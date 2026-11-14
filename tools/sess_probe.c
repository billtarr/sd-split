// fastrpc 会话探针：直接调 remote_session_control，不经过 llama.cpp
// 目的：在同一进程内连续申请 session 1 / session 2，记录原始 errno。
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <dlfcn.h>

#define FASTRPC_GET_EFFECTIVE_DOMAIN_ID 0x0000000e
#define FASTRPC_RESERVE_NEW_SESSION     0x00000002
#define FASTRPC_GET_URI                 0x0000000f

struct eff_domain {
    uint32_t domain_name_len;
    char    *domain_name;
    uint32_t session_id;
    uint32_t effective_domain_id;
};

struct reserve_sess {
    uint32_t domain_name_len;
    char    *domain_name;
    uint32_t session_name_len;
    char    *session_name;
    uint32_t session_id;
    uint32_t effective_domain_id;
};

typedef int (*rsc_t)(uint32_t req, void *data, uint32_t datalen);

static void meminfo(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return;
    char line[256];
    while (fgets(line, sizeof line, f)) {
        if (!strncmp(line, "MemAvailable", 12) || !strncmp(line, "MemTotal", 8))
            printf("  %s", line);
    }
    fclose(f);
}

int main(void) {
    printf("== meminfo ==\n");
    meminfo();

    void *h = dlopen("libcdsprpc.so", RTLD_NOW);
    if (!h) { printf("dlopen libcdsprpc.so failed: %s\n", dlerror()); return 1; }
    rsc_t rsc = (rsc_t) dlsym(h, "remote_session_control");
    if (!rsc) { printf("dlsym remote_session_control failed\n"); return 1; }

    const char *dom = "cdsp";

    struct eff_domain e;
    memset(&e, 0, sizeof e);
    e.domain_name     = (char *) dom;
    e.domain_name_len = (uint32_t) strlen(dom);
    e.session_id      = 0;
    int err = rsc(FASTRPC_GET_EFFECTIVE_DOMAIN_ID, &e, sizeof e);
    printf("GET_EFFECTIVE_DOMAIN_ID(\"%s\") = 0x%x, dom_id=%u\n", dom, err, e.effective_domain_id);

    for (uint32_t sid = 1; sid <= 2; sid++) {
        char name[32];
        snprintf(name, sizeof name, "probe-sess-%u", sid);
        struct reserve_sess r;
        memset(&r, 0, sizeof r);
        r.domain_name     = (char *) dom;
        r.domain_name_len = (uint32_t) strlen(dom);
        r.session_name    = name;
        r.session_name_len = (uint32_t) strlen(name);
        r.session_id      = sid;
        int er = rsc(FASTRPC_RESERVE_NEW_SESSION, &r, sizeof r);
        printf("RESERVE_NEW_SESSION sid=%u (%s) = 0x%x, eff_dom=%u\n", sid, name, er, r.effective_domain_id);
    }

    {
        struct get_uri {
            uint32_t session_id, domain_name_len; char *domain_name;
            uint32_t module_uri_len; char *module_uri;
            uint32_t uri_len; char *uri;
        } u;
        char uri[512] = {0};
        const char *mod = "libggml-htp-v73.so?htp_iface_skel_handle_invoke&_modver=1.0";
        memset(&u, 0, sizeof u);
        u.domain_name = (char *) dom; u.domain_name_len = (uint32_t) strlen(dom);
        u.module_uri = (char *) mod;  u.module_uri_len  = (uint32_t) strlen(mod);
        u.uri = uri; u.uri_len = sizeof uri;
        int er = rsc(FASTRPC_GET_URI, &u, sizeof u);
        printf("GET_URI(session 0, \"%s\") = 0x%x\n", dom, er);
    }

    printf("== meminfo after ==\n");
    meminfo();
    return 0;
}
