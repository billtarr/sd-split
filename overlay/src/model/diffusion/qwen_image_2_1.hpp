#ifndef __SD_MODEL_DIFFUSION_QWEN_IMAGE_2_1_H__
#define __SD_MODEL_DIFFUSION_QWEN_IMAGE_2_1_H__

#include "core/pe_cache.h"
#include "model/diffusion/qwen_image.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

namespace Qwen {

    // ---------------------------------------------------------------------------------------------
    // DiT 层区间切分（多进程分段的第一层，见 docs/npu-split.md）
    //
    // 目标：每段权重 < fastrpc 单 PD 的 2304 MB IOVA 窗口，于是段内一次性 map 到底、零搬运。
    // 这一层只做「按层区间跑 + joint 落盘」，不涉及 IPC；数值靠两次顺序跑（A 出 joint、B 吃 joint）
    // 与整模型逐 step 对账。IPC/常驻化是下一步。
    //
    //   SD_DIT_LAYER_FIRST=16 SD_DIT_LAYER_LAST=32
    //   SD_DIT_JOINT_IN=joint_<step>.bin      // first>0 时必须给
    //   SD_DIT_DUMP_DIR=dump/                 // 每 step 落 joint_<step>.bin / vel_<step>.bin
    // ---------------------------------------------------------------------------------------------

    inline int qwen_image_21_env_int(const char * name, int fallback) {
        const char * v = getenv(name);
        if (v == nullptr || *v == '\0') {
            return fallback;
        }
        return atoi(v);
    }

    inline std::string qwen_image_21_dump_path(const std::string & dir, const char * kind, int step) {
        char name[64];
        snprintf(name, sizeof(name), "/%s_%03d.bin", kind, step);
        return dir + name;
    }

    // int64 ndim | int64 dims[ndim] | float data[numel]
    inline bool qwen_image_21_dump_tensor(const std::string & path, const sd::Tensor<float>& t) {
        FILE * f = fopen(path.c_str(), "wb");
        if (f == nullptr) {
            LOG_WARN("qwen_image_2_1: cannot write %s", path.c_str());
            return false;
        }
        const int64_t nd = t.dim();
        bool ok = fwrite(&nd, sizeof(nd), 1, f) == 1 &&
                  fwrite(t.shape().data(), sizeof(int64_t), static_cast<size_t>(nd), f) == static_cast<size_t>(nd) &&
                  fwrite(t.data(), sizeof(float), static_cast<size_t>(t.numel()), f) == static_cast<size_t>(t.numel());
        fclose(f);
        return ok;
    }

    inline bool qwen_image_21_load_tensor(const std::string & path, sd::Tensor<float>& out) {
        FILE * f = fopen(path.c_str(), "rb");
        if (f == nullptr) {
            return false;
        }
        int64_t nd = 0;
        if (fread(&nd, sizeof(nd), 1, f) != 1 || nd <= 0 || nd > 8) {
            fclose(f);
            return false;
        }
        std::vector<int64_t> shape(static_cast<size_t>(nd));
        if (fread(shape.data(), sizeof(int64_t), static_cast<size_t>(nd), f) != static_cast<size_t>(nd)) {
            fclose(f);
            return false;
        }
        out.resize(shape);
        const bool ok = fread(out.data(), sizeof(float), static_cast<size_t>(out.numel()), f) ==
                        static_cast<size_t>(out.numel());
        fclose(f);
        return ok;
    }

    // ---------------------------------------------------------------------------------------------
    // 段间乒乓通道（见 docs/npu-split.md §5）
    //
    // 拓扑：末段（拥有采样器）→ 前一段 → … → 首段（拥有 prelude）。
    // 每步末段把 x 往下发，首段算完 prelude + 自己的层把 joint 往上回，中间的段接力。
    // 全同步、无队列：发送方写完才读，所以天然就是 lockstep，不需要步号对齐逻辑。
    // 走 socketpair（AF_UNIX），4.3 MB 的 joint 一趟是内核态拷贝，量级几毫秒。
    // ---------------------------------------------------------------------------------------------

    inline bool qwen_image_21_write_all(int fd, const void * p, size_t n) {
        const uint8_t * q = static_cast<const uint8_t *>(p);
        while (n > 0) {
            const ssize_t k = ::write(fd, q, n);
            if (k <= 0) {
                return false;
            }
            q += k;
            n -= static_cast<size_t>(k);
        }
        return true;
    }

    inline bool qwen_image_21_read_all(int fd, void * p, size_t n) {
        uint8_t * q = static_cast<uint8_t *>(p);
        while (n > 0) {
            const ssize_t k = ::read(fd, q, n);
            if (k <= 0) {
                return false;
            }
            q += k;
            n -= static_cast<size_t>(k);
        }
        return true;
    }

    // 帧格式：int32 step | int32 ndim | uint32 timestep_bits | int64 dims[ndim] | float32 data[numel]
    //
    // timestep 必须跟着 latent 一起走：段的输入里 timestep 会进 modulation，中间的段没有采样器
    // （常驻 worker 见 serve_split_worker），自己推不出这一步的 t。（CLI 模式下各进程步数/调度
    // 相同，以前是各算各的；现在统一以末段发来的为准，NaN 表示「未知，用调用方自己的」。）
    inline uint32_t qwen_image_21_ts_bits(float ts) {
        uint32_t bits = 0;
        std::memcpy(&bits, &ts, sizeof(bits));
        return bits;
    }

    inline bool qwen_image_21_send_tensor(int fd, int step, const sd::Tensor<float>& t, float ts = std::nanf("")) {
        const int32_t hdr[3] = { step, static_cast<int32_t>(t.dim()), static_cast<int32_t>(qwen_image_21_ts_bits(ts)) };
        return qwen_image_21_write_all(fd, hdr, sizeof(hdr)) &&
               qwen_image_21_write_all(fd, t.shape().data(), sizeof(int64_t) * static_cast<size_t>(t.dim())) &&
               qwen_image_21_write_all(fd, t.data(), sizeof(float) * static_cast<size_t>(t.numel()));
    }

    inline bool qwen_image_21_recv_tensor(int fd, int * step, sd::Tensor<float>& t, float * ts = nullptr) {
        int32_t hdr[3] = { 0, 0, 0 };
        if (!qwen_image_21_read_all(fd, hdr, sizeof(hdr))) {
            return false;
        }
        if (hdr[1] <= 0 || hdr[1] > 8) {
            return false;
        }
        const uint32_t ts_bits = static_cast<uint32_t>(hdr[2]);
        std::vector<int64_t> shape(static_cast<size_t>(hdr[1]));
        if (!qwen_image_21_read_all(fd, shape.data(), sizeof(int64_t) * shape.size())) {
            return false;
        }
        if (step != nullptr) {
            *step = hdr[0];
        }
        if (ts != nullptr) {
            float ts_value = 0.f;
            std::memcpy(&ts_value, &ts_bits, sizeof(ts_value));
            *ts = ts_value;
        }
        t.resize(shape);
        return qwen_image_21_read_all(fd, t.data(), sizeof(float) * static_cast<size_t>(t.numel()));
    }

    struct QwenImage21Config {
        int64_t in_channels       = 64;
        int64_t out_channels      = 64;
        int64_t hidden_size       = 4096;
        int64_t context_dim       = 4096;
        int64_t head_dim          = 128;
        int64_t intermediate_size = 12288;
        int num_layers            = 32;
        bool fused_mlp            = false;
        std::vector<int> axes_dim = {16, 56, 56};

        static QwenImage21Config detect_from_weights(const String2TensorStorage& weights, const std::string& prefix) {
            QwenImage21Config config;
            auto find = [&](const std::string& suffix) -> const TensorStorage* {
                auto it = weights.find(prefix + "." + suffix);
                return it == weights.end() ? nullptr : &it->second;
            };
            if (auto w = find("img_in.weight")) {
                config.in_channels = w->logical_ne0();
                config.hidden_size = w->ne[1];
            }
            if (auto w = find("proj_out.weight")) {
                config.out_channels = w->ne[1];
            }
            if (auto w = find("txt_in.in_layer.weight")) {
                config.context_dim = w->logical_ne0();
            }
            if (auto w = find("transformer_blocks.0.attn.norm_q.weight")) {
                config.head_dim = w->ne[0];
            }
            if (auto w = find("transformer_blocks.0.img_mlp.gate_up.weight")) {
                config.intermediate_size = w->ne[1] / 2;
                config.fused_mlp         = true;
            } else if (auto w = find("transformer_blocks.0.img_mlp.proj.weight")) {
                config.intermediate_size = w->ne[1];
            }
            int layers                     = 0;
            const std::string block_prefix = prefix + ".transformer_blocks.";
            for (const auto& [name, _] : weights) {
                if (starts_with(name, block_prefix)) {
                    layers = std::max(layers, atoi(name.substr(block_prefix.size()).c_str()) + 1);
                }
            }
            if (layers > 0) {
                config.num_layers = layers;
                LOG_VERBOSE("qwen_image_2_1: layers = %d, hidden_size = %" PRId64 ", context_dim = %" PRId64,
                            layers, config.hidden_size, config.context_dim);
            }
            return config;
        }
    };

    struct QwenImage21Segment {
        int64_t start;
        int64_t end;
        int64_t context_start;
        int image_index;
    };

    struct QwenImage21Layout {
        std::vector<QwenImage21Segment> segments;
        std::vector<std::vector<float>> positions;
        int64_t prefix_length = 0;
        Rope::PositionLayout rope_layout;

        static QwenImage21Layout build(int64_t text_length,
                                       const sd::Tensor<int32_t>& image_slots,
                                       const std::vector<std::pair<int64_t, int64_t>>& image_shapes) {
            if (image_shapes.empty() || (!image_slots.empty() && image_slots.numel() != text_length)) {
                throw std::runtime_error("Qwen Image 2.1: invalid image token layout");
            }
            QwenImage21Layout layout;
            int64_t position  = 0;
            int next_image    = 0;
            auto append_image = [&](int index, int64_t context_start) {
                auto [height, width] = image_shapes[index];
                int64_t start        = static_cast<int64_t>(layout.positions.size());
                layout.segments.push_back({start, start + height * width, context_start, index});
                layout.rope_layout.append_image(static_cast<int>(height), static_cast<int>(width));
                for (int64_t h = 0; h < height; ++h) {
                    for (int64_t w = 0; w < width; ++w) {
                        layout.positions.push_back({static_cast<float>(position),
                                                    static_cast<float>(h - (height - height / 2)),
                                                    static_cast<float>(w - (width - width / 2))});
                    }
                }
                position += std::max(height, width);
            };
            for (int64_t i = 0; i < text_length;) {
                int tag       = image_slots.empty() ? 0 : image_slots[i];
                int64_t begin = i++;
                while (i < text_length && (image_slots.empty() ? 0 : image_slots[i]) == tag) {
                    ++i;
                }
                if (tag != 0) {
                    if (tag != next_image + 1 || next_image + 1 >= static_cast<int>(image_shapes.size()) ||
                        (i - begin) * 4 != image_shapes[next_image].first * image_shapes[next_image].second) {
                        throw std::runtime_error("Qwen Image 2.1: vision slots and reference latents must have matching sizes");
                    }
                    append_image(next_image++, begin);
                } else {
                    int64_t start = static_cast<int64_t>(layout.positions.size());
                    layout.segments.push_back({start, start + i - begin, begin, -1});
                    layout.rope_layout.append_tokens(i - begin);
                    for (int64_t j = begin; j < i; ++j, ++position) {
                        float p = static_cast<float>(position);
                        layout.positions.push_back({p, p, p});
                    }
                }
            }
            if (next_image + 1 != static_cast<int>(image_shapes.size())) {
                throw std::runtime_error("Qwen Image 2.1: missing reference image slots");
            }
            layout.prefix_length = static_cast<int64_t>(layout.positions.size());
            append_image(next_image, text_length);
            return layout;
        }
    };

    struct QwenImage21PrefixCache {
        enum class Mode {
            NONE,
            STORE,
            REUSE
        };
        Mode mode = Mode::NONE;
        std::string name;
        std::string cut_group;
        int64_t prefix_length = 0;
        ggml_type type        = GGML_TYPE_F32;
        bool* flash_attn_used = nullptr;
    };

    class QwenImage21ZeroCenterRMSNorm : public RMSNorm {
    public:
        using RMSNorm::RMSNorm;

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
            auto weight = params["weight"];
            if (ctx->weight_adapter) {
                weight = ctx->weight_adapter->patch_weight(ctx, weight, prefix + "weight");
            }
            weight = ggml_scale_bias(ctx->ggml_ctx, weight, 1.f, 1.f);
            return ggml_mul(ctx->ggml_ctx, ggml_rms_norm(ctx->ggml_ctx, x, eps), weight);
        }
    };

    class QwenImage21TextProjection : public GGMLBlock {
    public:
        QwenImage21TextProjection(const QwenImage21Config& config) {
            blocks["text_norm"] = std::make_shared<QwenImage21ZeroCenterRMSNorm>(config.context_dim, 1e-6f);
            blocks["in_layer"]  = std::make_shared<Linear>(config.context_dim, config.hidden_size, false);
            blocks["out_layer"] = std::make_shared<Linear>(config.hidden_size, config.hidden_size, false);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            x = std::dynamic_pointer_cast<QwenImage21ZeroCenterRMSNorm>(blocks["text_norm"])->forward(ctx, x);
            x = std::dynamic_pointer_cast<Linear>(blocks["in_layer"])->forward(ctx, x);
            x = ggml_ext_gelu(ctx, x);
            return std::dynamic_pointer_cast<Linear>(blocks["out_layer"])->forward(ctx, x);
        }
    };

    class QwenImage21Attention : public QwenImageAttention {
    public:
        QwenImage21Attention(const QwenImage21Config& config)
            : QwenImageAttention(config.hidden_size, config.head_dim, config.hidden_size / config.head_dim, 0, 0, false, false) {
            for (const auto* name : {"add_q_proj", "add_k_proj", "add_v_proj", "norm_added_q", "norm_added_k", "to_add_out"}) {
                blocks.erase(name);
            }
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* pe, const std::vector<QwenImage21Segment>& segments, const std::vector<ggml_tensor*>& masks, const QwenImage21PrefixCache& cache) {
            int64_t heads = x->ne[0] / dim_head;
            auto project  = [&](const char* name) {
                auto h = std::dynamic_pointer_cast<Linear>(blocks[name])->forward(ctx, x);
                return ggml_reshape_4d(ctx->ggml_ctx, h, dim_head, heads, x->ne[1], x->ne[2]);
            };
            auto q = project("to_q");
            auto k = project("to_k");
            auto v = project("to_v");
            q      = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_q"])->forward(ctx, q);
            k      = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_k"])->forward(ctx, k);
            q      = Rope::apply_rope(ctx, q, pe);
            k      = Rope::apply_rope(ctx, k, pe);
            if (cache.mode == QwenImage21PrefixCache::Mode::STORE) {
                // Preserve query-first attention evaluation while writing each layer's
                // prefix before its full-sequence K/V can accumulate across layers.
                ctx->expand_graph(q);
                auto persist = [&](ggml_tensor* tensor, int axis, const char* name) {
                    auto part = ggml_ext_slice(ctx, tensor, axis, 0, cache.prefix_length);
                    // Pack the contiguous data into wider rows so quantization blocks
                    // can exceed head_dim without padding or changing element order.
                    part      = ggml_reshape_2d(ctx->ggml_ctx, part, x->ne[0], cache.prefix_length);
                    auto copy = ggml_cast(ctx->ggml_ctx, part, cache.type);
                    // Keep the copy in this layer's segment so graph cuts do not
                    // retain or recompute the full-sequence K/V in the final segment.
                    sd::ggml_graph_cut::mark_graph_cut(copy, cache.cut_group, name);
                    ctx->persist_cache_tensor(cache.name + "." + name, copy);
                };
                persist(k, 1, "k");
                persist(v, 2, "v");
            }
            auto attend = [&](ggml_tensor* aq, ggml_tensor* ak, ggml_tensor* av, ggml_tensor* mask) {
                bool used_flash_attn = false;
                auto out             = ggml_ext_attention_ext(ctx, aq, ak, av, heads, mask, true, ctx->flash_attn_enabled, 1.f, &used_flash_attn);
                if (cache.flash_attn_used != nullptr) {
                    *cache.flash_attn_used &= used_flash_attn;
                }
                return out;
            };
            ggml_tensor* result = nullptr;
            if (cache.mode == QwenImage21PrefixCache::Mode::REUSE) {
                auto prefix_k = ctx->load_cache_tensor(cache.name + ".k");
                auto prefix_v = ctx->load_cache_tensor(cache.name + ".v");
                GGML_ASSERT(prefix_k != nullptr && prefix_v != nullptr);
                if (prefix_k->type != k->type) {
                    prefix_k = ggml_cast(ctx->ggml_ctx, prefix_k, k->type);
                }
                if (prefix_v->type != v->type) {
                    prefix_v = ggml_cast(ctx->ggml_ctx, prefix_v, v->type);
                }
                prefix_k = ggml_reshape_4d(ctx->ggml_ctx, prefix_k, dim_head, cache.prefix_length, heads, k->ne[3]);
                prefix_v = ggml_reshape_4d(ctx->ggml_ctx, prefix_v, dim_head, heads, cache.prefix_length, v->ne[3]);
                k        = ggml_concat(ctx->ggml_ctx, prefix_k, k, 1);
                v        = ggml_concat(ctx->ggml_ctx, prefix_v, v, 2);
                result   = attend(q, k, v, nullptr);
            } else {
                for (size_t i = 0; i < segments.size(); ++i) {
                    const auto& segment = segments[i];
                    auto sq             = ggml_ext_slice(ctx, q, 1, segment.start, segment.end);
                    auto sk             = ggml_ext_slice(ctx, k, 1, 0, segment.end);
                    auto sv             = ggml_ext_slice(ctx, v, 2, 0, segment.end);
                    auto out            = attend(sq, sk, sv, masks[i]);
                    result              = result == nullptr ? out : ggml_concat(ctx->ggml_ctx, result, out, 1);
                }
            }
            auto to_out = std::dynamic_pointer_cast<Linear>(blocks["to_out.0"]);
            to_out->set_precision_flags(GGML_PRECISION_F32_VULKAN | GGML_PRECISION_F32_ROCM);
            return to_out->forward(ctx, result);
        }
    };

    class QwenImage21TransformerBlock : public GGMLBlock {
    public:
        QwenImage21TransformerBlock(const QwenImage21Config& config) {
            blocks["img_norm1"] = std::make_shared<LayerNorm>(config.hidden_size, 1e-6f, false);
            blocks["img_norm2"] = std::make_shared<LayerNorm>(config.hidden_size, 1e-6f, false);
            blocks["attn"]      = std::make_shared<QwenImage21Attention>(config);
            if (config.fused_mlp) {
                blocks["img_mlp.gate_up"] = std::make_shared<Linear>(config.hidden_size, 2 * config.intermediate_size, false);
            } else {
                blocks["img_mlp.proj"]       = std::make_shared<Linear>(config.hidden_size, config.intermediate_size, false);
                blocks["img_mlp.gate_layer"] = std::make_shared<Linear>(config.hidden_size, config.intermediate_size, false);
            }
            blocks["img_mlp.out"] = std::make_shared<Linear>(config.intermediate_size, config.hidden_size, false);
        }

        static ggml_tensor* modulate(GGMLRunnerContext* runner_ctx, ggml_tensor* x, ggml_tensor* params, int64_t prefix_length, bool gate = false) {
            auto ctx   = runner_ctx->ggml_ctx;
            auto rows  = ggml_ext_chunk(runner_ctx, params, 2, 1);
            auto apply = [&](ggml_tensor* part, ggml_tensor* row) {
                row = gate ? ggml_tanh(ctx, row) : ggml_scale_bias(ctx, row, 1.f, 1.f);
                runner_ctx->expand_graph(row);
                return ggml_mul(ctx, part, row);
            };
            if (prefix_length == 0) {
                return apply(x, rows[0]);
            }
            auto target = apply(ggml_ext_slice(runner_ctx, x, 1, prefix_length, x->ne[1]), rows[0]);
            auto prefix = apply(ggml_ext_slice(runner_ctx, x, 1, 0, prefix_length), rows[1]);
            return ggml_concat(ctx, prefix, target, 1);
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, const std::vector<ggml_tensor*>& modulation, ggml_tensor* pe, const QwenImage21Layout& layout, const std::vector<ggml_tensor*>& masks, const QwenImage21PrefixCache& cache) {
            const int64_t prefix_length = cache.mode == QwenImage21PrefixCache::Mode::REUSE ? 0 : layout.prefix_length;
            auto h                      = std::dynamic_pointer_cast<LayerNorm>(blocks["img_norm1"])->forward(ctx, x);
            h                           = modulate(ctx, h, modulation[0], prefix_length);
            h                           = std::dynamic_pointer_cast<QwenImage21Attention>(blocks["attn"])->forward(ctx, h, pe, layout.segments, masks, cache);
            x                           = ggml_add(ctx->ggml_ctx, x, modulate(ctx, h, modulation[1], prefix_length, true));
            h                           = std::dynamic_pointer_cast<LayerNorm>(blocks["img_norm2"])->forward(ctx, x);
            h                           = modulate(ctx, h, modulation[2], prefix_length);
            auto fused                  = blocks.find("img_mlp.gate_up");
            if (fused != blocks.end()) {
                auto gate_up = std::dynamic_pointer_cast<Linear>(fused->second)->forward(ctx, h);
                h            = ggml_swiglu(ctx->ggml_ctx, gate_up);
            } else {
                auto gate = std::dynamic_pointer_cast<Linear>(blocks["img_mlp.gate_layer"])->forward(ctx, h);
                h         = std::dynamic_pointer_cast<Linear>(blocks["img_mlp.proj"])->forward(ctx, h);
                h         = ggml_swiglu_split(ctx->ggml_ctx, gate, h);
            }
            h = std::dynamic_pointer_cast<Linear>(blocks["img_mlp.out"])->forward(ctx, h);
            return ggml_add(ctx->ggml_ctx, x, modulate(ctx, h, modulation[3], prefix_length, true));
        }
    };

    class QwenImage21Model : public GGMLBlock {
        QwenImage21Config config;

    public:
        // Layer range this segment owns: [layer_first, layer_last).  last < num_layers means this
        // segment ends before the head (norm_out/proj_out), so its graph output is `joint`.
        int layer_first = 0;
        int layer_last  = 0;
        bool is_split() const { return layer_first > 0 || layer_last < config.num_layers; }

    private:
        void apply_layer_range() {
            const int first = qwen_image_21_env_int("SD_DIT_LAYER_FIRST", 0);
            const int last  = qwen_image_21_env_int("SD_DIT_LAYER_LAST", static_cast<int>(config.num_layers));
            layer_first     = std::clamp(first, 0, static_cast<int>(config.num_layers));
            layer_last      = std::clamp(last, layer_first, static_cast<int>(config.num_layers));
        }

    public:
        QwenImage21Model(const QwenImage21Config& config)
            : config(config) {
            apply_layer_range();
            blocks["time_text_embed.timestep_embedder"] = std::make_shared<TimestepEmbedding>(256, config.hidden_size, 0, 0, false);
            blocks["txt_in"]                            = std::make_shared<QwenImage21TextProjection>(config);
            blocks["img_in"]                            = std::make_shared<Linear>(config.in_channels, config.hidden_size, false);
            blocks["modulation.1"]                      = std::make_shared<Linear>(config.hidden_size, 4 * config.hidden_size, false);
            blocks["norm_out.linear"]                   = std::make_shared<Linear>(config.hidden_size, config.hidden_size, false);
            blocks["norm_out.norm"]                     = std::make_shared<LayerNorm>(config.hidden_size, 1e-6f, false);
            blocks["proj_out"]                          = std::make_shared<Linear>(config.hidden_size, config.out_channels, false);
            // Only the blocks this segment owns are created, so their tensors are never requested
            // from the gguf and never allocated on any backend.
            for (int i = layer_first; i < layer_last; ++i) {
                blocks["transformer_blocks." + std::to_string(i)] = std::make_shared<QwenImage21TransformerBlock>(config);
            }
            if (is_split()) {
                LOG_WARN("qwen_image_2_1: layer split active, this segment owns layers [%d, %d) of %d",
                         layer_first, layer_last, static_cast<int>(config.num_layers));
            }
        }

        // joint_in != nullptr: this segment starts mid-stack, so the prelude (img_in/txt_in/refs) is
        // skipped and `joint` is taken from the caller.  When layer_last < num_layers the function
        // returns `joint` before the prefix slice / norm_out / proj_out.
        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* timestep, ggml_tensor* context, const std::vector<ggml_tensor*>& refs, ggml_tensor* pe, const QwenImage21Layout& layout, const std::vector<ggml_tensor*>& masks, const QwenImage21PrefixCache& cache, ggml_tensor* joint_in = nullptr) {
            auto time = ggml_concat(ctx->ggml_ctx, timestep, ggml_ext_zeros_like(ctx, timestep), 0);
            // Runtime flow timesteps already use the [0, 1000] scale.
            time               = ggml_ext_timestep_embedding(ctx, time, 256, 10000, 1.f);
            time               = std::dynamic_pointer_cast<TimestepEmbedding>(blocks["time_text_embed.timestep_embedder"])->forward(ctx, time);
            time               = ggml_silu(ctx->ggml_ctx, time);
            auto modulation    = std::dynamic_pointer_cast<Linear>(blocks["modulation.1"])->forward(ctx, time);
            auto mod           = ggml_ext_chunk(ctx, modulation, 4, 0);
            auto img_in        = std::dynamic_pointer_cast<Linear>(blocks["img_in"]);
            ggml_tensor* joint = nullptr;
            if (joint_in != nullptr) {
                // Mid-stack segment: the prelude lives in a previous segment.
                joint = joint_in;
            } else if (cache.mode == QwenImage21PrefixCache::Mode::REUSE) {
                joint = img_in->forward(ctx, DiT::patchify(ctx, x, 1, 1));
            } else {
                auto text = std::dynamic_pointer_cast<QwenImage21TextProjection>(blocks["txt_in"])->forward(ctx, context);
                for (const auto& segment : layout.segments) {
                    ggml_tensor* h;
                    if (segment.image_index < 0) {
                        h = ggml_ext_slice(ctx, text, 1, segment.context_start,
                                           segment.context_start + segment.end - segment.start);
                    } else {
                        auto image = segment.image_index == static_cast<int>(refs.size()) ? x : refs[segment.image_index];
                        h          = img_in->forward(ctx, DiT::patchify(ctx, image, 1, 1));
                    }
                    joint = joint == nullptr ? h : ggml_concat(ctx->ggml_ctx, joint, h, 1);
                }
            }
            if (joint_in == nullptr) {
                sd::ggml_graph_cut::mark_graph_cut(joint, "qwen_image_2_1.prelude", "joint");
            }
            for (int i = layer_first; i < layer_last; ++i) {
                const std::string layer = "transformer_blocks." + std::to_string(i);
                auto layer_cache        = cache;
                layer_cache.name        = cache.name + "." + std::to_string(i);
                layer_cache.cut_group   = "qwen_image_2_1." + layer;
                auto block              = std::dynamic_pointer_cast<QwenImage21TransformerBlock>(blocks[layer]);
                joint                   = block->forward(ctx, joint, mod, pe, layout, masks, layer_cache);
                sd::ggml_graph_cut::mark_graph_cut(joint, layer_cache.cut_group, "joint");
            }
            if (layer_last < config.num_layers) {
                // Intermediate segment: `joint` (prefix included) is the graph output.
                return joint;
            }
            if (cache.mode != QwenImage21PrefixCache::Mode::REUSE) {
                joint = ggml_ext_slice(ctx, joint, 1, layout.prefix_length, joint->ne[1]);
            }
            auto scale = std::dynamic_pointer_cast<Linear>(blocks["norm_out.linear"])->forward(ctx, ggml_ext_chunk(ctx, time, 2, 1)[0]);
            joint      = std::dynamic_pointer_cast<LayerNorm>(blocks["norm_out.norm"])->forward(ctx, joint);
            joint      = ggml_mul(ctx->ggml_ctx, joint, ggml_scale_bias(ctx->ggml_ctx, scale, 1.f, 1.f));
            joint      = std::dynamic_pointer_cast<Linear>(blocks["proj_out"])->forward(ctx, joint);
            return DiT::unpatchify_and_crop(ctx, joint, x->ne[1], x->ne[0], 1, 1);
        }
    };

    struct QwenImage21Runner : public DiffusionModelRunner {
        sd::PECache pe_cache_;
        QwenImage21Config config;
        QwenImage21Model model;
        std::vector<sd::Tensor<float>> mask_data;
        ggml_type prefix_cache_type = GGML_TYPE_COUNT;
        bool prefix_cache_enabled   = true;
        bool prefix_cache_disabled  = false;
        bool prefix_cache_auto_f32  = false;
        int  split_step_            = 0;   // index of the sampling step this process is on

        // 链式分段（见 docs/npu-split.md §5）
        bool  split_init_       = false;
        bool  split_worker_     = false;  // 有后继：本进程是别人的前段
        int   split_next_fd_    = -1;     // 到后继
        int   split_prev_fd_    = -1;     // 到前驱（本进程 spawn 出来的）
        int   split_steps_      = 0;      // 总步数（子进程跑完就退，省掉无意义的 VAE 解码）
        pid_t split_child_pid_  = -1;
        // context 交接：worker 不加载 TE（省 ~4.8 GB/进程，见 I-029），文本条件由末段写一次文件给它。
        bool          split_ctx_sent_ = false;
        const char *  split_ctx_file  = nullptr;
        sd::Tensor<float> split_ctx_hold_;        // worker 缓存的 context（只读一次文件）
        bool              split_ctx_hold_valid_ = false;
        // worker 从末段收到这一步的 timestep（进 modulation 用）。NaN = 没收到，退回自己的采样器。
        float             split_ts_recv_ = std::nanf("");
        // 一次性跑（sd-cli）为 true：worker 服务完最后一步就退。服务模式（sd-server）设 0，
        // worker 常驻，靠 step 回到 0 判定新请求。
        bool              split_exit_ = true;

        // SD_DIT_SPLIT_PLAN="11,22" → 边界 [0,11) [11,22) [22,32)
        static std::vector<int> split_plan() {
            std::vector<int> out;
            const char * v = getenv("SD_DIT_SPLIT_PLAN");
            if (v == nullptr || *v == '\0') {
                return out;
            }
            const char * p = v;
            while (*p != '\0') {
                char * end = nullptr;
                const long b = strtol(p, &end, 10);
                if (end == p) {
                    break;
                }
                out.push_back(static_cast<int>(b));
                p = *end == ',' ? end + 1 : end;
            }
            return out;
        }

        // 以 /proc/self/cmdline 原样重启自己，只改层区间与 fd；子进程会成为我们的前驱。
        void split_spawn_predecessor(int first, int last) {
            std::vector<std::string> args;
            {
                std::ifstream f("/proc/self/cmdline", std::ios::binary);
                std::string   cur;
                char          c = 0;
                while (f.get(c)) {
                    if (c == '\0') {
                        args.push_back(cur);
                        cur.clear();
                    } else {
                        cur.push_back(c);
                    }
                }
                if (!cur.empty()) {
                    args.push_back(cur);
                }
            }
            if (args.empty()) {
                LOG_ERROR("qwen_image_2_1: split cannot read /proc/self/cmdline");
                return;
            }
            // 子进程跑的是同一份 argv，会跟着写同一个 -o 文件（它的图是占位 velocity 出的垃圾），
            // 后写完的会覆盖掉真正的结果 → 把子进程的输出改到 /dev/null。
            for (size_t i = 0; i + 1 < args.size(); ++i) {
                if (args[i] == "-o" || args[i] == "--output") {
                    args[i + 1] = "/dev/null";
                    break;
                }
            }

            int sv[2] = { -1, -1 };
            if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) {
                LOG_ERROR("qwen_image_2_1: split socketpair failed");
                return;
            }

            const pid_t pid = fork();
            if (pid == 0) {
                // fork 会把父进程的 fd 表整份复制过去，ggml-hexagon 的 dspqueue / rpcmem 句柄
                // 都没有 CLOEXEC。exec 后子进程若还持有它们，就会和父进程的 PD 撞车。
                // 只留 stdout/stderr 和这一段 socketpair。
                for (int fd = 3; fd < 1024; ++fd) {
                    if (fd != sv[0]) {
                        ::close(fd);
                    }
                }
                setenv("SD_DIT_LAYER_FIRST", std::to_string(first).c_str(), 1);
                setenv("SD_DIT_LAYER_LAST", std::to_string(last).c_str(), 1);
                setenv("SD_DIT_SPLIT_FD", std::to_string(sv[0]).c_str(), 1);
                // 子进程是 worker：不加载 TE/VAE，文本条件由本进程写 ctx 文件交给它（I-029）
                setenv("SD_DIT_SPLIT_WORKER", "1", 1);
                if (split_ctx_file != nullptr) {
                    setenv("SD_DIT_CTX_FILE", split_ctx_file, 1);
                }
                setenv("SD_DIT_DUMP_DIR", "", 1);
                unsetenv("SD_DIT_X_IN");
                unsetenv("SD_DIT_JOINT_IN");
                std::vector<char *> av;
                av.reserve(args.size() + 1);
                for (auto & a : args) {
                    av.push_back(a.data());
                }
                av.push_back(nullptr);
                execv("/proc/self/exe", av.data());
                _exit(127);
            }
            if (pid < 0) {
                LOG_ERROR("qwen_image_2_1: split fork failed");
                ::close(sv[0]);
                ::close(sv[1]);
                return;
            }

            ::close(sv[0]);
            split_prev_fd_    = sv[1];
            split_child_pid_  = pid;
            LOG_WARN("qwen_image_2_1: split spawned predecessor pid %d for layers [%d, %d)", (int) pid, first, last);
        }

        void split_setup(bool from_init = false) {
            if (split_init_) {
                return;
            }
            // ★ 默认只在「第一次 compute()」时 spawn。把 spawn 提到 model.init() 以让子进程和父进程
            //   的 TE 并行，实测**每次都把整机搞挂**（三进程几乎同时做 HTP init）：加了 fd 清理和
            //   6 s 错峰也一样挂。这是个禁止项，不是待调参数。见 docs/npu-split.md §5。
            //   SD_DIT_SPLIT_EARLY=1 仅用于复现该故障，不要在生产里开。
            if (from_init && getenv("SD_DIT_SPLIT_EARLY") == nullptr) {
                return;
            }
            split_init_ = true;

            if (const char * st = getenv("SD_DIT_SPLIT_STEPS"); st != nullptr && *st != '\0') {
                split_steps_ = atoi(st);
            }
            // context 交接文件（末段写、worker 读）。只在 worker 里真正被读到。
            if (const char * ex = getenv("SD_DIT_SPLIT_EXIT"); ex != nullptr && *ex != '\0') {
                split_exit_ = atoi(ex) != 0;
            }
            split_ctx_file = getenv("SD_DIT_CTX_FILE");
            if (split_ctx_file == nullptr || *split_ctx_file == '\0') {
                split_ctx_file = "qs-split-ctx.bin";
            }

            if (const char * fd_env = getenv("SD_DIT_SPLIT_FD"); fd_env != nullptr && *fd_env != '\0') {
                split_next_fd_ = atoi(fd_env);
                split_worker_  = true;
                // 不要再漏给我们的子进程
                fcntl(split_next_fd_, F_SETFD, FD_CLOEXEC);
                LOG_WARN("qwen_image_2_1: split worker, successor fd %d", split_next_fd_);
            }

            const bool file_mode = getenv("SD_DIT_JOINT_IN") != nullptr;
            if (model.layer_first > 0 && !file_mode) {
                const auto plan = split_plan();
                int        prev = 0;
                bool       found = false;
                for (size_t i = 0; i < plan.size(); ++i) {
                    if (plan[i] == model.layer_first) {
                        prev  = i == 0 ? 0 : plan[i - 1];
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    LOG_ERROR("qwen_image_2_1: SD_DIT_LAYER_FIRST=%d not found in SD_DIT_SPLIT_PLAN",
                              model.layer_first);
                    return;
                }
                // 中间段错开一点再 spawn 自己的前驱：三个进程同一瞬间做 HTP init（各建一个 PD、
                // 各映射上万兆）实测会把整机搞挂（I-023 同一类故障）。
                if (split_worker_) {
                    const char * stg = getenv("SD_DIT_SPLIT_STAGGER_MS");
                    const int    ms  = stg != nullptr && *stg != '\0' ? atoi(stg) : 6000;
                    if (ms > 0) {
                        usleep(static_cast<useconds_t>(ms) * 1000);
                    }
                }
                split_spawn_predecessor(prev, model.layer_first);
            }
        }

        void split_shutdown() {
            if (split_prev_fd_ >= 0) {
                ::close(split_prev_fd_);
                split_prev_fd_ = -1;
            }
            if (split_child_pid_ > 0) {
                int st = 0;
                waitpid(split_child_pid_, &st, 0);
                split_child_pid_ = -1;
            }
        }

        static bool supports_prefix_cache_type(ggml_type type) {
            if (type == GGML_TYPE_F32) {
                return true;
            }
            const auto* traits = ggml_get_type_traits(type);
            if (traits->from_float_ref == nullptr || traits->to_float == nullptr) {
                return false;
            }
            auto cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            if (cpu == nullptr) {
                return false;
            }
            auto ctx = std::unique_ptr<ggml_context, decltype(&ggml_free)>(
                ggml_init({3 * ggml_tensor_overhead(), nullptr, true}), ggml_free);
            if (ctx == nullptr) {
                return false;
            }
            // Some reference quantizers have no runtime copy support. Query the
            // device through the registry so dynamically loaded CPU backends work.
            auto source  = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, ggml_blck_size(type));
            auto encoded = ggml_cast(ctx.get(), source, type);
            auto decoded = ggml_cast(ctx.get(), encoded, GGML_TYPE_F32);
            return ggml_backend_dev_supports_op(cpu, encoded) && ggml_backend_dev_supports_op(cpu, decoded);
        }

        QwenImage21Runner(ggml_backend_t backend, const String2TensorStorage& weights, const std::string& prefix, std::shared_ptr<RunnerWeightManager> weight_manager = nullptr, const char* model_args = nullptr)
            : DiffusionModelRunner(backend, prefix, weight_manager),
              config(QwenImage21Config::detect_from_weights(weights, prefix)),
              model(config) {
            for (const auto& [key, value] : parse_key_value_args(model_args, "model arg")) {
                if (key == "qwen_image_2_1_prefix_cache" && !parse_strict_bool(value, prefix_cache_enabled)) {
                    LOG_WARN("ignoring invalid Qwen Image 2.1 model arg '%s=%s'", key.c_str(), value.c_str());
                } else if (key == "qwen_image_2_1_prefix_cache_type") {
                    if (value == "auto") {
                        prefix_cache_type = GGML_TYPE_COUNT;
                        continue;
                    }
                    const auto type = sd_type_to_ggml_type(str_to_sd_type(value.c_str()));
                    if (type == GGML_TYPE_COUNT) {
                        LOG_WARN("ignoring unknown Qwen Image 2.1 cache type '%s'", value.c_str());
                    } else if (!supports_prefix_cache_type(type)) {
                        LOG_WARN("ignoring Qwen Image 2.1 cache type '%s': runtime conversion to and from F32 is unavailable", value.c_str());
                    } else if (config.hidden_size % ggml_blck_size(type) != 0) {
                        LOG_WARN("ignoring Qwen Image 2.1 cache type '%s': block size %" PRId64 " does not divide hidden size %" PRId64,
                                 value.c_str(), ggml_blck_size(type), config.hidden_size);
                    } else {
                        prefix_cache_type = type;
                    }
                }
            }
            model.init(params_ctx, weights, prefix);
            // 在这里就把前驱启起来，让它和本进程的 TE 并行。放到 compute() 里会晚一步：
            // 那时父进程 TE 已经跑完，子进程的加载 + TE 全部串行加到第一步前面（实测 +58 s）。
            split_setup(true);
        }

        std::string get_desc() override { return "qwen_image_2_1"; }

        // 常驻 worker 驱动入口（sd-server 用，见 docs/npu-split.md §9）。
        //
        // 不跑采样器：只重复「收 latent(+timestep) → compute → 发 joint」。x / context / timestep
        // 全部来自 socket，所以本循环对分辨率、步数、prompt 一无所知也不会错位，换分辨率也不用重建。
        // 之所以能脱离采样器，是因为 timestep 现在随 latent 一起下发（见帧格式），段的 modulation 不再依赖
        // 「各进程各自跑一遍同样的调度」这个隐含前提。
        bool serve_split_worker(int n_threads) override {
            split_setup();  // 读 SD_DIT_SPLIT_FD / 拉起本段的前驱
            if (!split_worker_) {
                LOG_ERROR("qwen_image_2_1: serve_split_worker requires SD_DIT_SPLIT_FD");
                return false;
            }
            sd::Tensor<float> timestep = sd::Tensor<float>::from_vector({ 0.f });
            DiffusionParams   params;
            params.timesteps = &timestep;  // 占位：真正用的是 socket 收到的 timestep
            params.extra     = QwenImage21DiffusionExtra{};
            LOG_WARN("qwen_image_2_1: split worker serving layers [%d, %d)", model.layer_first, model.layer_last);
            for (;;) {
                auto out = compute(n_threads, params);
                if (out.empty()) {
                    // 末段关闭 fd（或段内报错）→ 本段使命结束。直接 _exit：不要再跑 HTP 的析构。
                    LOG_WARN("qwen_image_2_1: split worker loop ended");
                    fflush(nullptr);
                    _exit(0);
                }
            }
            return true;
        }

        // 末段每次请求结束都要重发 context：常驻 worker 不会自己重读文件，也不会重新算 TE。
        void reset_split_state() override {
            if (!split_worker_ && split_prev_fd_ >= 0) {
                split_ctx_sent_ = false;
            }
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) override {
            model.get_param_tensors(tensors, prefix);
        }

        bool has_prefix_cache(const QwenImage21PrefixCache& cache) {
            for (int i = 0; i < config.num_layers; ++i) {
                const auto name = cache.name + "." + std::to_string(i);
                auto k          = get_cache_tensor_by_name(name + ".k");
                auto v          = get_cache_tensor_by_name(name + ".v");
                if (k == nullptr || v == nullptr || k->type != cache.type || v->type != cache.type ||
                    k->ne[0] != config.hidden_size || k->ne[1] != cache.prefix_length || k->ne[2] != 1 || k->ne[3] != 1 ||
                    v->ne[0] != config.hidden_size || v->ne[1] != cache.prefix_length || v->ne[2] != 1 || v->ne[3] != 1) {
                    return false;
                }
            }
            return true;
        }

        sd::Tensor<float> compute(int n_threads, const DiffusionParams& inputs) override {
            split_setup();

            sd::Tensor<float> x_local   = tensor_or_empty(inputs.x);
            sd::Tensor<float> ctx_local = tensor_or_empty(inputs.context);
            const auto& x       = x_local;
            const auto& context = ctx_local;

            // 作为别人的前段：这一步的 latent 由后继发来，采样器给的 x 作废。
            if (split_worker_) {
                int               rstep = split_step_;
                sd::Tensor<float> rx;
                float             rts = std::nanf("");
                for (;;) {
                    if (!qwen_image_21_recv_tensor(split_next_fd_, &rstep, rx, &rts)) {
                        LOG_ERROR("qwen_image_2_1: split worker recv latent failed (step %d)", split_step_);
                        return {};
                    }
                    if (rstep < 0) {
                        // 控制帧：末段每次请求的第一步前发一次 context（`(-1, ctx)`）。
                        // 走 socket 而不是文件 —— 提前 spawn 时 worker 会先于末段算完 TE 就绪，
                        // 读文件必然读到上一次请求的旧 ctx（实测 `[4096,16]` → invalid image token layout）。
                        split_ctx_hold_       = rx;
                        split_ctx_hold_valid_ = true;
                        LOG_WARN("qwen_image_2_1: split ctx recv %s numel %lld (via socket)",
                                 sd::tensor_shape_to_string(rx.shape()).c_str(), (long long) rx.numel());
                        if (split_prev_fd_ >= 0 &&
                            !qwen_image_21_send_tensor(split_prev_fd_, -1, rx)) {
                            LOG_ERROR("qwen_image_2_1: split ctx forward failed");
                            return {};
                        }
                        continue;
                    }
                    break;
                }
                if (!x.empty() && rx.shape() != x.shape()) {
                    LOG_ERROR("qwen_image_2_1: split worker latent shape mismatch %s vs %s",
                              sd::tensor_shape_to_string(rx.shape()).c_str(),
                              sd::tensor_shape_to_string(x.shape()).c_str());
                    return {};
                }
                x_local        = std::move(rx);
                split_step_    = rstep;
                split_ts_recv_ = rts;
            }
            // 段内 timestep：worker 一律用末段发来的 t（它没有采样器，自己推不出这一步的 t）。
            sd::Tensor<float>       ts_hold;
            const sd::Tensor<float>* ts_ptr = inputs.timesteps;
            if (split_worker_ && std::isfinite(split_ts_recv_)) {
                ts_hold = sd::Tensor<float>::from_vector({ split_ts_recv_ });
                ts_ptr  = &ts_hold;
            }
            if (split_worker_ && split_ctx_hold_valid_) {
                ctx_local = split_ctx_hold_;
            } else if (split_worker_ && split_ctx_file != nullptr && *split_ctx_file != '\0') {
                // worker 没加载 TE：文本条件一律以末段写的文件为准。不能只在 `inputs.context` 为空时
                // 才读 —— 没有 conditioner 时上游仍会塞一个维度不对的占位 context（实测 [3584,9]，
                // 而 DiT 要 4096），直接用会让整个 worker 在这里失败。
                if (!split_ctx_hold_valid_) {
                    if (qwen_image_21_load_tensor(split_ctx_file, split_ctx_hold_)) {
                        split_ctx_hold_valid_ = true;
                        LOG_WARN("qwen_image_2_1: ctx loaded %s numel %lld from %s",
                                 sd::tensor_shape_to_string(split_ctx_hold_.shape()).c_str(),
                                 (long long) split_ctx_hold_.numel(), split_ctx_file);
                    } else {
                        LOG_ERROR("qwen_image_2_1: split worker cannot read context '%s'", split_ctx_file);
                    }
                }
                if (split_ctx_hold_valid_) {
                    ctx_local = split_ctx_hold_;
                }
            }

            if (x.empty() || context.empty() || context.dim() < 2 || context.shape()[0] != config.context_dim ||
                tensor_or_empty(ts_ptr).numel() != 1 ||
                x.dim() != 4 || x.shape()[3] != 1 || x.shape()[2] != config.in_channels) {
                LOG_ERROR("Qwen Image 2.1 requires an image latent and text conditioning with batch size 1 "
                          "(x %s ctx %s ctx-dim %d ctx-ne0 %lld ts %lld in_ch %d)",
                          x.empty() ? "empty" : sd::tensor_shape_to_string(x.shape()).c_str(),
                          context.empty() ? "empty" : sd::tensor_shape_to_string(context.shape()).c_str(),
                          config.context_dim,
                          context.empty() ? -1LL : (long long) context.shape()[0],
                          (long long) tensor_or_empty(ts_ptr).numel(),
                          config.in_channels);
                return {};
            }
            static const std::vector<sd::Tensor<float>> empty_refs;
            const auto& refs = inputs.ref_latents && inputs.ref_image_params.pass_to_dit ? *inputs.ref_latents : empty_refs;
            std::vector<std::pair<int64_t, int64_t>> shapes;
            for (const auto& ref : refs) {
                if (ref.dim() != 4 || ref.shape()[2] != config.in_channels || ref.shape()[3] != 1) {
                    LOG_ERROR("Qwen Image 2.1: invalid reference latent shape");
                    return {};
                }
                shapes.emplace_back(ref.shape()[1], ref.shape()[0]);
            }
            shapes.emplace_back(x.shape()[1], x.shape()[0]);
            const auto* extra = std::get_if<QwenImage21DiffusionExtra>(&inputs.extra);
            QwenImage21Layout layout;
            try {
                layout = QwenImage21Layout::build(context.shape()[1], tensor_or_empty(extra ? extra->image_slots : nullptr), shapes);
            } catch (const std::exception& error) {
                LOG_ERROR("%s", error.what());
                return {};
            }
            if (!runner_started()) {
                prefix_cache_disabled = false;
                prefix_cache_auto_f32 = false;
            }
            // ---- layer-split bookkeeping (see docs/npu-split.md) ----------------------------------
            // SD_DIT_X_IN      : dir holding x_<step>.bin, overrides the sampler's latent for replay.
            // SD_DIT_JOINT_IN  : dir holding joint_<step>.bin produced by the previous segment.
            // SD_DIT_DUMP_DIR  : dir to write x_/joint_/vel_<step>.bin into.
            const char* split_x_in     = getenv("SD_DIT_X_IN");
            const char* split_joint_in = getenv("SD_DIT_JOINT_IN");
            const char* split_dump_dir = getenv("SD_DIT_DUMP_DIR");
            if (split_x_in != nullptr && *split_x_in != '\0') {
                sd::Tensor<float> injected;
                const auto path = qwen_image_21_dump_path(split_x_in, "x", split_step_);
                if (!qwen_image_21_load_tensor(path, injected)) {
                    LOG_ERROR("qwen_image_2_1: cannot load injected latent %s", path.c_str());
                    return {};
                }
                if (injected.shape() != x.shape()) {
                    LOG_ERROR("qwen_image_2_1: injected latent shape mismatch (%s vs %s)",
                              sd::tensor_shape_to_string(injected.shape()).c_str(),
                              sd::tensor_shape_to_string(x.shape()).c_str());
                    return {};
                }
                x_local = std::move(injected);
            }
            sd::Tensor<float> joint_in_data;
            if (split_prev_fd_ >= 0) {
                // 链式模式：把这一步的 latent 发给前驱，拿回 joint。
                if (!split_ctx_sent_ && model.layer_last >= static_cast<int>(config.num_layers)) {
                    // 第一步之前先把文本条件作为控制帧 `(-1, ctx)` 沿链发下去（147 KB）。
                    // 早先写文件的做法在「提前 spawn」下必挂：worker 就绪早于本进程算完 TE，
                    // 会读到上一次的旧 ctx。走 socket 后由 recv 天然定序。
                    if (!context.empty()) {
                        if (!qwen_image_21_send_tensor(split_prev_fd_, -1, context)) {
                            LOG_ERROR("qwen_image_2_1: split send ctx failed");
                            return {};
                        }
                        LOG_WARN("qwen_image_2_1: ctx sent %s numel %lld (via socket)",
                                 sd::tensor_shape_to_string(context.shape()).c_str(),
                                 (long long) context.numel());
                    }
                    split_ctx_sent_ = true;
                }
                // timestep 跟着 latent 一起下发：中间段没有采样器，自己推不出这一步的 t。
                const float send_ts = tensor_or_empty(ts_ptr).numel() == 1 ? ts_ptr->data()[0] : std::nanf("");
                if (!qwen_image_21_send_tensor(split_prev_fd_, split_step_, x, send_ts)) {
                    LOG_ERROR("qwen_image_2_1: split send latent failed (step %d)", split_step_);
                    return {};
                }
                int jstep = split_step_;
                if (!qwen_image_21_recv_tensor(split_prev_fd_, &jstep, joint_in_data)) {
                    LOG_ERROR("qwen_image_2_1: split recv joint failed (step %d)", split_step_);
                    return {};
                }
            } else if (model.layer_first > 0) {
                if (split_joint_in == nullptr || *split_joint_in == '\0') {
                    LOG_ERROR("qwen_image_2_1: SD_DIT_LAYER_FIRST > 0 requires SD_DIT_JOINT_IN or a split predecessor");
                    return {};
                }
                const auto path = qwen_image_21_dump_path(split_joint_in, "joint", split_step_);
                if (!qwen_image_21_load_tensor(path, joint_in_data)) {
                    LOG_ERROR("qwen_image_2_1: cannot load injected joint %s", path.c_str());
                    return {};
                }
            }
            if (split_dump_dir != nullptr && *split_dump_dir != '\0') {
                qwen_image_21_dump_tensor(qwen_image_21_dump_path(split_dump_dir, "x", split_step_), x);
            }
            // A mid-stack segment cannot own the prefix cache (its K/V spans the whole stack).
            const bool split_tail = model.layer_last < static_cast<int>(config.num_layers);
            auto finalize = [&](sd::Tensor<float> out) -> sd::Tensor<float> {
                if (out.empty()) {
                    return out;
                }
                if (split_dump_dir != nullptr && *split_dump_dir != '\0') {
                    qwen_image_21_dump_tensor(qwen_image_21_dump_path(split_dump_dir, split_tail ? "joint" : "vel", split_step_), out);
                }
                const bool tail = split_tail || split_worker_;
                if (split_worker_) {
                    // 把 joint 交回后继；本进程的采样器拿到的是占位 velocity，反正它出的图不用。
                    if (!qwen_image_21_send_tensor(split_next_fd_, split_step_, out)) {
                        LOG_ERROR("qwen_image_2_1: split send joint failed (step %d)", split_step_);
                        return {};
                    }
                }
                split_step_++;
                if (split_steps_ > 0 && split_step_ >= split_steps_) {
                    // 服务模式：步号回绕，让下一轮请求的 worker 能识别「新请求」
                    split_step_ = split_exit_ ? split_step_ : 0;
                }
                if (tail) {
                    // 子进程服务完最后一步就没用了。它的采样器只会产出垃圾图，VAE 解码纯属
                    // 和真正的末段抢 CPU（实测三个并发解码把端到端从 ~113 s 拖到 206 s）。
                    if (split_worker_ && split_exit_ && split_steps_ > 0 && split_step_ >= split_steps_) {
                        fflush(nullptr);
                        _exit(0);
                    }
                    return sd::Tensor<float>(x.shape());
                }
                return out;
            };
            // ---------------------------------------------------------------------------------------
            QwenImage21PrefixCache cache;
            if (!model.is_split() && prefix_cache_enabled && !prefix_cache_disabled && extra != nullptr && extra->prefix_id != 0 && layout.prefix_length > 0) {
                cache.name          = "qwen_image_2_1.prefix." + std::to_string(extra->prefix_id) +
                                      ".circular." + std::to_string(circular_x_enabled) + std::to_string(circular_y_enabled);
                cache.prefix_length = layout.prefix_length;
                if (prefix_cache_type != GGML_TYPE_COUNT) {
                    cache.type = prefix_cache_type;
                } else if (!prefix_cache_auto_f32 && flash_attn_enabled && !sage_attn_enabled &&
                           (attn_scale <= 0.f || attn_scale == 1.f)) {
                    cache.type = GGML_TYPE_F16;
                }
                cache.mode = has_prefix_cache(cache) ? QwenImage21PrefixCache::Mode::REUSE : QwenImage21PrefixCache::Mode::STORE;
            }
            bool flash_attn_used = true;
            auto run             = [&](const QwenImage21PrefixCache& active_cache) {
                flash_attn_used    = true;
                auto checked_cache = active_cache;
                if (prefix_cache_type == GGML_TYPE_COUNT && active_cache.type == GGML_TYPE_F16) {
                    checked_cache.flash_attn_used = &flash_attn_used;
                }
                const bool cached = active_cache.mode == QwenImage21PrefixCache::Mode::REUSE;
                mask_data.clear();
                if (!cached) {
                    for (const auto& segment : layout.segments) {
                        sd::Tensor<float> mask;
                        if (segment.image_index < 0) {
                            mask = sd::Tensor<float>::zeros({segment.end, segment.end - segment.start});
                            for (int64_t q = segment.start; q < segment.end; ++q) {
                                for (int64_t k = q + 1; k < segment.end; ++k) {
                                    mask[k + segment.end * (q - segment.start)] = -INFINITY;
                                }
                            }
                        }
                        mask_data.push_back(std::move(mask));
                    }
                }
                auto build = [&]() {
                    auto graph = new_graph_custom(QWEN_IMAGE_GRAPH_SIZE * 2);
                    sd::PECacheKey pe_key("qwen_image_2_1.pe");
                    pe_key.add(layout.positions).add(cached).add(layout.prefix_length).add(config.axes_dim).add(config.head_dim);
                    pe_key.add(circular_x_enabled).add(circular_y_enabled);
                    for (const auto& region : layout.rope_layout.images) {
                        pe_key.add(region.begin).add(region.count).add(region.height_period).add(region.width_period);
                        pe_key.add(region.height_axis).add(region.width_axis);
                    }
                    auto pe = make_input(pe_cache_.get_or_create(pe_key, [&]() {
                        const auto first_position = layout.positions.begin() + (cached ? layout.prefix_length : 0);
                        Rope::Embedding embedding;
                        embedding.ids.assign(first_position, layout.positions.end());
                        const size_t offset             = cached ? static_cast<size_t>(layout.prefix_length) : 0;
                        embedding.positions.token_count = embedding.ids.size();
                        for (auto region : layout.rope_layout.images) {
                            if (region.begin >= offset) {
                                region.begin -= offset;
                                embedding.positions.images.push_back(region);
                            }
                        }
                        embedding.values     = Rope::embed_nd(embedding.ids, 1, 10000.f, config.axes_dim, embedding.layout, &embedding.frequencies);
                        auto values          = finish_rope_pe(std::move(embedding));
                        const int64_t tokens = static_cast<int64_t>(layout.positions.size()) - (cached ? layout.prefix_length : 0);
                        return sd::Tensor<float>({2, 2, config.head_dim / 2, tokens}, std::move(values));
                    }));
                    std::vector<ggml_tensor*> masks, ref_inputs;
                    for (const auto& mask : mask_data) {
                        masks.push_back(mask.empty() ? nullptr : make_input(mask));
                    }
                    ggml_tensor* joint_t = joint_in_data.empty() ? nullptr : make_input(joint_in_data);
                    if (!cached && joint_t == nullptr) {
                        for (const auto& ref : refs) {
                            ref_inputs.push_back(make_input(ref));
                        }
                    }
                    auto ctx = get_context(graph);
                    auto out = model.forward(&ctx, make_input(x), make_input(*ts_ptr), cached ? nullptr : make_input(context),
                                             ref_inputs, pe, layout, masks, checked_cache, joint_t);
                    if (!flash_attn_used) {
                        return static_cast<ggml_cgraph*>(nullptr);
                    }
                    ggml_build_forward_expand(graph, out);
                    return graph;
                };
                return finalize(restore_trailing_singleton_dims(GGMLRunner::compute(build, n_threads, false), x.dim()));
            };
            auto result = run(cache);
            if (result.empty() && !flash_attn_used) {
                // Casting an F16 cache back to F32 cannot recover its original values.
                // Recompute the prefix before executing a graph that falls back from FA.
                free_cache_ctx_and_buffer();
                prefix_cache_auto_f32 = true;
                cache.type            = GGML_TYPE_F32;
                cache.mode            = QwenImage21PrefixCache::Mode::STORE;
                LOG_DEBUG("Qwen Image 2.1: Flash Attention unavailable; using F32 prefix caching for this sampling run");
                result = run(cache);
            }
            if (result.empty() && last_compute_status() == GGML_STATUS_ALLOC_FAILED &&
                (cache.mode != QwenImage21PrefixCache::Mode::NONE || has_cache_tensors())) {
                // The failed graph has ended before persistent inputs are released.
                free_cache_ctx_and_buffer();
                prefix_cache_disabled = true;
                LOG_WARN("Qwen Image 2.1: insufficient memory for prefix caching; retrying without it for this sampling run");
                return run(QwenImage21PrefixCache{});
            }
            if (!result.empty() && cache.mode == QwenImage21PrefixCache::Mode::STORE) {
                if (!has_prefix_cache(cache)) {
                    free_cache_ctx_and_buffer();
                    prefix_cache_disabled = true;
                    LOG_WARN("Qwen Image 2.1: incomplete prefix cache; disabling it for this sampling run");
                } else {
                    LOG_DEBUG("Qwen Image 2.1: cached prefix %" PRIu64 " (%" PRId64 " tokens, %s)", extra->prefix_id, layout.prefix_length, ggml_type_name(cache.type));
                }
            }
            return result;
        }
    };
}

#endif  // __SD_MODEL_DIFFUSION_QWEN_IMAGE_2_1_H__
