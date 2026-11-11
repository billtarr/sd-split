/*
 * Minimal stand-in for the Hexagon SDK <remote.h>.
 *
 * The types must match the SDK byte for byte (they cross into libcdsprpc.so),
 * and the prototypes are implemented in htp-drv.cpp as wrappers over dlsym()'d
 * entry points.
 *
 * ABIFIXME: the FASTRPC_* request codes and the fastrpc_domain layout are
 * reconstructed from the SDK documentation rather than copied out of the SDK.
 * They only feed remote_system_request(FASTRPC_GET_DOMAINS) -- the optional
 * "auto-discover the CDSP domain id" path -- plus the virtual-session request.
 * A mismatch there makes discovery fail and the code falls back to the
 * configured/default domain id; the compute path does not depend on them.
 */
#pragma once

#include "AEEStdDef.h"
#include "domain.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t remote_handle;
typedef uint64_t remote_handle64;

typedef union remote_arg {
    void * pv;
    int    fd;
    struct {
        void *   pv;    /* buffer address -- FIRST field, confirmed by disassembly */
        uint32_t nLen;  /* buffer length */
    } buf;
} remote_arg;

#if 0
/* (previous, wrong order -- kept only as a note of what the disassembly ruled out:
 * the SDK stores pv at offset 0 and nLen at offset 8, not the other way round) */
struct { uint32_t nLen; void * pv; } buf;
#endif

/*
 * fastrpc scalar encoding: method id in the low 6 bits, then 4 bits each for
 * in-buffers / in-pointers / out-buffers / out-pointers, packet type on top.
 * The generated qaic stub bakes these constants into its method table, so this
 * macro must match the SDK exactly or every IDL call is rejected at run time
 * (ioctl ENOTTY / 0xe).
 *
 * Recovered from llama-hex's prebuilt libggml-hexagon.so, where the compiler
 * folded the literals:
 *   htp_iface_hwinfo  MAKEX(0, 8, 0, 1, 0, 0) -> 0x08000100  (id<<24 | in_ptrs<<8)
 *   htp_iface_start   MAKEX(0, 2, 1, 0, 0, 0) -> 0x02010000  (id<<24 | in_bufs<<16)
 *   htp_iface_mmap    MAKEX(0, 4, 1, 0, 0, 0) -> 0x04010000
 * So: id in the top byte, then in_bufs, in_ptrs, and the out counts below.
 * The out slots are never exercised by this backend (always 0) -- their exact
 * position is inferred, not confirmed.
 */
#define REMOTE_SCALARS_MAKE(id, in_bufs, in_ptrs, out_bufs, out_ptrs) \
    ((((id)       & 0xff) << 24) | (((in_bufs)  & 0xf) << 16) | \
     (((in_ptrs)  & 0xf)  <<  8) | (((out_bufs) & 0xf) <<  4) | \
     (((out_ptrs) & 0xf)  <<  0))

#define REMOTE_SCALARS_MAKEX(pkt, id, in_bufs, in_ptrs, out_bufs, out_ptrs) \
    ((((pkt) & 0x3) << 30) | REMOTE_SCALARS_MAKE(id, in_bufs, in_ptrs, out_bufs, out_ptrs))

/*
 * These are bit flags, not an ordinal enum -- the three values ggml-hexagon
 * actually passes were read out of the compiled ternary in
 * ggml_hexagon_shared_buffer::mmap() (w5 = pinned ? 2 : extended ? 18 : 3):
 *    FASTRPC_MAP_FD                  = 2
 *    FASTRPC_MAP_FD_DELAYED          = 3   (FD | delay)
 *    FASTRPC_MAP_FD_DELAYED_EXTENDED = 18  (FD | 16)
 * Passing the slow value here fails the ioctl with ENOTTY (0xe).
 */
enum fastrpc_map_flags {
    FASTRPC_MAP_FD                  = 2,
    FASTRPC_MAP_FD_DELAYED          = 3,
    FASTRPC_MAP_FD_NOMAP            = 6,   /* FD | 4, not exercised */
    FASTRPC_MAP_FD_DELAYED_EXTENDED = 18,
    FASTRPC_MAP_FD_UNMAP            = 0,
};

/* Domain kinds reported by FASTRPC_GET_DOMAINS. */
enum fastrpc_domain_type {
    FASTRPC_UNKNOWN_DOMAIN = 0,
    FASTRPC_CDSP           = 1,
    FASTRPC_ADSP           = 2,
    FASTRPC_NSP            = 3,  /* what the NPU shows up as */
    FASTRPC_MDSP           = 4,
    FASTRPC_SDSP           = 5,
};

#define FASTRPC_DOMAIN_NAME_LEN 256

typedef struct fastrpc_domain {
    int32   type;
    int32   id;
    char    name[FASTRPC_DOMAIN_NAME_LEN];
    int32   status;
    int32   instance_id;
} fastrpc_domain;

/*
 * Request ids. These are NOT guessable and NOT in the same namespace:
 * remote_session_control() and remote_handle64_control() each have their own
 * numbering, which is why GET_DSP_INFO and CONTROL_UNSIGNED_MODULE are both 2.
 *
 * Recovered by disassembling llama-hex's prebuilt libggml-hexagon.so and reading
 * the immediate loaded into w0 before each call -- the only reliable way without
 * the SDK headers. A wrong value here shows up as ioctl ENOTTY (0x14).
 */
#define FASTRPC_GET_DOMAINS             0x00000001
#define FASTRPC_RESERVE_NEW_SESSION     0x00000002
#define FASTRPC_SET_PATH                0x00000003
#define FASTRPC_GET_EFFECTIVE_DOMAIN_ID 0x0000000e  /* 14, confirmed */
#define FASTRPC_GET_URI                 0x0000000f  /* 15, confirmed */
#define FASTRPC_MANAGE_DSP_LIBRARY_PATH 0x00000014  /* 20, confirmed */

/* remote_handle_control() request ids. */
#define DSPRPC_GET_DSP_INFO              0x00000002  /* confirmed, sizeof payload 12 */
#define DSPRPC_CONTROL_LATENCY           0x00000001  /* confirmed, sizeof payload 8  */
#define DSPRPC_CONTROL_UNSIGNED_MODULE   0x00000002  /* confirmed, sizeof payload 8  */
#define DSPRPC_SET_MODE                  0x00000003

/* remote_dsp_capability::attribute_ID values. */
#define ARCH_VER 0

struct remote_dsp_capability {
    uint32_t domain;
    uint32_t attribute_ID;
    uint32_t capability;
};

/*
 * Control payloads for remote_session_control() / remote_handle64_control().
 *
 * ABIFIXME: field order below is reconstructed, not copied from the SDK. Each
 * one is checked at runtime and has a fallback path: GET_URI falls back to a
 * hand-built URI string, GET_EFFECTIVE_DOMAIN_ID falls back to the configured
 * domain id, and DSPRPC_CONTROL_LATENCY / UNSIGNED_MODULE only warn (QLOG) on
 * failure. None of them sit in the compute path.
 */
struct remote_rpc_effective_domain_id {
    char *   domain_name;
    uint32_t domain_name_len;
    uint32_t session_id;
    uint32_t effective_domain_id;
};

struct remote_rpc_control_unsigned_module {
    int32_t domain;
    int32_t enable;
};

struct remote_rpc_get_uri {
    uint32_t session_id;
    uint32_t domain_name_len;
    char *   domain_name;
    uint32_t module_uri_len;
    char *   module_uri;
    uint32_t uri_len;
    char *   uri;
};

struct remote_rpc_control_latency {
    int32_t enable;
    int32_t latency;
};

/* Multi-session reservation (only used when a device exposes several virtual
 * sessions of the same CDSP core). */
struct remote_rpc_reserve_new_session {
    uint32_t domain_name_len;
    char *   domain_name;
    uint32_t session_name_len;
    char *   session_name;
    uint32_t session_id;
    uint32_t effective_domain_id;
};

/* FASTRPC_MANAGE_DSP_LIBRARY_PATH payload. */
struct fastrpc_manage_dsp_lib_path {
    int32_t effec_domain_id;
    int32_t type;
    char *  path;
    int32_t pathlen;
};

/* The default CDSP (NPU) domain on Qualcomm SoCs. */
#define CDSP_DOMAIN_ID   3
#define CDSP_DOMAIN_NAME "cdsp"
/* DOMAINS list filter, packed as (count << 8) | type. */
#define DOMAINS_LIST_FLAGS_SET_TYPE(count, type) (((uint32) (count) << 8) | ((uint32) (type) & 0xff))

typedef struct system_req_payload {
    uint32 id;
    union {
        struct {
            fastrpc_domain * domains;
            int32            max_domains;
            int32            num_domains;
            uint32           flags;
        } sys;
    };
} system_req_payload;

/* Implemented in htp-drv.cpp over dlsym()'d libcdsprpc entry points. */
int fastrpc_mmap  (int domain, int fd, void * addr, int offset, size_t length, enum fastrpc_map_flags flags);
int fastrpc_munmap(int domain, int fd, void * addr, size_t length);

int remote_handle64_open   (const char * name, remote_handle64 * ph);
int remote_handle64_invoke (remote_handle64 h, uint32_t dwScalars, remote_arg * pra);
int remote_handle64_close  (remote_handle64 h);
int remote_handle_control  (uint32_t req, void * data, uint32_t datalen);
int remote_handle64_control(remote_handle64 h, uint32_t req, void * data, uint32_t datalen);
int remote_session_control (uint32_t req, void * data, uint32_t datalen);
int remote_system_request  (system_req_payload * req);

#ifdef __cplusplus
}
#endif
