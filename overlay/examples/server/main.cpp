#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <sched.h>
#include <string>
#include <thread>
#include <vector>

#include "httplib.h"

#include "async_jobs.h"
#include "common/common.h"
#include "common/resource_owners.hpp"
#include "model_catalog.h"
#include "routes.h"
#include "runtime.h"

#ifdef HAVE_INDEX_HTML
#include "frontend/dist/gen_index_html.h"
#endif

// CPU 亲和性：`SD_CPU_AFFINITY=f8`（十六进制掩码，bit i = cpu i）。
// 为什么要在进程内做而不是外面套 taskset：APK 里 exe 是被 ProcessBuilder 直接拉起的，
// 中间没有 shell 可以插 taskset。而 sched_setaffinity 会被 fork/exec 的子进程继承，
// 所以在这里设一次，spawn 出来的分段 worker 全部跟着绑——实测 sampling 也快了 6%
// （host 侧线程不再被 A510 小核拖）。
static void apply_cpu_affinity() {
    const char* m = std::getenv("SD_CPU_AFFINITY");
    if (m == nullptr || *m == '\0') {
        return;
    }
    const unsigned long mask = std::strtoul(m, nullptr, 16);
    cpu_set_t           set;
    CPU_ZERO(&set);
    for (int i = 0; i < 32; ++i) {
        if (mask & (1UL << i)) {
            CPU_SET(i, &set);
        }
    }
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        LOG_WARN("SD_CPU_AFFINITY=%s failed: %s", m, std::strerror(errno));
    } else {
        LOG_INFO("cpu affinity mask 0x%lx applied", mask);
    }
}

static void print_usage(const char* argv0, const std::vector<ArgOptions>& options_list) {
    std::cout << version_string() << "\n";
    std::cout << "Usage: " << argv0 << " [options]\n\n";
    std::cout << "Svr Options:\n";
    options_list[0].print();
    std::cout << "\nContext Options:\n";
    options_list[1].print();
    std::cout << "\nDefault Generation Options:\n";
    options_list[2].print();
}

static void parse_args(int argc,
                       const char** argv,
                       SDSvrParams& svr_params,
                       SDContextParams& ctx_params,
                       SDGenerationParams& default_gen_params) {
    std::vector<ArgOptions> options_vec = {
        svr_params.get_options(),
        ctx_params.get_options(),
        default_gen_params.get_options(),
    };

    if (!parse_options(argc, argv, options_vec)) {
        print_usage(argv[0], options_vec);
        exit(svr_params.normal_exit ? 0 : 1);
    }

    log_level = svr_params.log_level;
    log_color = svr_params.color;

    if (ctx_params.list_devices) {
        exit(ctx_params.print_devices() ? 0 : 1);
    }

    const bool random_seed_requested = default_gen_params.seed < 0;

    if (!svr_params.resolve_and_validate() ||
        !ctx_params.resolve_and_validate(ctx_params.model_path.empty() && ctx_params.diffusion_model_path.empty() ? METADATA : IMG_GEN) ||
        !default_gen_params.resolve_and_validate(IMG_GEN,
                                                 ctx_params.lora_model_dir,
                                                 ctx_params.hires_upscalers_dir)) {
        print_usage(argv[0], options_vec);
        exit(1);
    }

    if (random_seed_requested) {
        default_gen_params.seed = -1;
    }
}

void sd_log_cb(enum sd_log_level_t level, const char* log, void* data) {
    SDSvrParams* svr_params = (SDSvrParams*)data;
    log_print(level, log, svr_params->log_level, svr_params->color);
}

int main(int argc, const char** argv) {
    apply_cpu_affinity();
    if (argc > 1 && std::string(argv[1]) == "--version") {
        std::cout << version_string() << "\n";
        return EXIT_SUCCESS;
    }
    SDSvrParams svr_params;
    SDContextParams ctx_params;
    // Preserve the CLI default while disabling cwd scans for sd-server.
    ctx_params.lora_model_dir.clear();
    SDGenerationParams default_gen_params;

    sd_set_log_callback(sd_log_cb, (void*)&svr_params);
    parse_args(argc, argv, svr_params, ctx_params, default_gen_params);

    LOG_VERBOSE("version: %s", version_string().c_str());
    LOG_VERBOSE("%s", sd_get_system_info());
    LOG_DEBUG("%s", svr_params.to_string().c_str());
    LOG_DEBUG("%s", ctx_params.to_string().c_str());
    LOG_DEBUG("%s", default_gen_params.to_string().c_str());

    sd_ctx_params_t sd_ctx_params = ctx_params.to_sd_ctx_params_t();
    ServerModels models;
    if (!ctx_params.model_path.empty() || !ctx_params.diffusion_model_path.empty()) {
        const auto started = std::chrono::steady_clock::now();
        models.context.reset(new_sd_ctx(&sd_ctx_params));
        models.status["load_elapsed_seconds"] = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    }

    if ((!ctx_params.model_path.empty() || !ctx_params.diffusion_model_path.empty()) && !models.context) {
        LOG_ERROR("new_sd_ctx_t failed");
        return 1;
    }

    // ---- 分段 worker 分发 ------------------------------------------------------------------------
    // spawn 用的是 `execv("/proc/self/exe", 原 argv)`（见 qwen_image_2_1.hpp），所以本进程被拉起时
    // 拿的是同一份 sd-server 命令行。这里必须抢在起 HTTP 监听之前分流，否则 worker 也会去 listen
    // （端口占用 + 永远走不到 compute()，末段死等，见 I-045）。
    // worker 只跑「收 latent → 算本段 → 发 joint」，不碰采样器/TE/VAE。
    if (const char* worker_env = std::getenv("SD_DIT_SPLIT_WORKER"); worker_env != nullptr && std::atoi(worker_env) != 0) {
        LOG_WARN("split worker mode: layers [%s, %s), no HTTP listener",
                 std::getenv("SD_DIT_LAYER_FIRST") ? std::getenv("SD_DIT_LAYER_FIRST") : "0",
                 std::getenv("SD_DIT_LAYER_LAST") ? std::getenv("SD_DIT_LAYER_LAST") : "-");
        if (!models.context) {
            LOG_ERROR("split worker needs a loaded diffusion model");
            return 1;
        }
        if (!sd_serve_split_worker(models.context.get())) {
            LOG_ERROR("split worker loop failed");
            return 1;
        }
        return 0;
    }

    // ---- TE 预热 ----------------------------------------------------------------------------------
    // Qwen3-VL-8B Q4_K_M 在 CPU 上第一次编码要 22~33 s，之后 4~7 s（同一进程连跑实测 22.65→7.54→5.58）。
    // 这不是随机抖动，是「首次」的一次性开销：权重页缺页 + CPU 从 idle 升频 + 大块 buffer 首次触碰。
    // 在 listen 之前先编两次，把这段开销挪到还没人发请求的窗口里。SD_TE_WARMUP=0 关掉，=N 指定轮数。
    if (models.context) {
        const char* warm = std::getenv("SD_TE_WARMUP");
        const int   rounds = warm == nullptr ? 2 : std::atoi(warm);
        if (rounds > 0) {
            if (!sd_warmup_conditioner(models.context.get(), rounds)) {
                LOG_WARN("TE warmup failed; the first request will pay the cold-start cost");
            }
        }
    }

    std::mutex sd_ctx_mutex;

    std::vector<LoraEntry> lora_cache;
    std::mutex lora_mutex;
    std::vector<UpscalerEntry> upscaler_cache;
    std::mutex upscaler_mutex;
    AsyncJobManager async_job_manager;
    async_job_manager.max_pending_jobs =
        static_cast<size_t>(svr_params.max_pending_jobs);
    async_job_manager.completed_ttl_seconds =
        static_cast<int64_t>(svr_params.completed_ttl_seconds);
    ServerRuntime runtime = {
        models.context.get(),
        &sd_ctx_mutex,
        &svr_params,
        &ctx_params,
        &default_gen_params,
        &lora_cache,
        &lora_mutex,
        &upscaler_cache,
        &upscaler_mutex,
        &async_job_manager,
        &models,
    };

    try {
        initialize_server_models(runtime);
    } catch (const std::exception& e) {
        LOG_ERROR("model catalog initialization failed: %s", e.what());
        return 1;
    }

    std::thread async_worker(async_job_worker, std::ref(runtime));

    httplib::Server svr;

    svr.set_pre_routing_handler([](const httplib::Request& req, httplib::Response& res) {
        std::string origin = req.get_header_value("Origin");
        if (origin.empty()) {
            origin = "*";
        }
        res.set_header("Access-Control-Allow-Origin", origin);
        res.set_header("Access-Control-Allow-Credentials", "true");
        res.set_header("Access-Control-Allow-Methods", "*");
        res.set_header("Access-Control-Allow-Headers", "*");

        if (req.method == "OPTIONS") {
            res.status = 204;
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    std::string index_html;
#ifdef HAVE_INDEX_HTML
    index_html.assign(reinterpret_cast<const char*>(index_html_bytes), index_html_size);
#else
    index_html = "Stable Diffusion Server is running";
#endif
    register_index_endpoints(svr, svr_params, index_html);
    register_openai_api_endpoints(svr, runtime);
    register_sdapi_endpoints(svr, runtime);
    register_sdcpp_api_endpoints(svr, runtime);
    register_model_endpoints(svr, runtime);

    LOG_INFO("listening on: http://%s:%d\n", svr_params.listen_ip.c_str(), svr_params.listen_port);
    svr.listen(svr_params.listen_ip, svr_params.listen_port);

    {
        std::lock_guard<std::mutex> lock(async_job_manager.mutex);
        async_job_manager.stop = true;
    }
    async_job_manager.cv.notify_all();
    async_worker.join();
    return 0;
}
