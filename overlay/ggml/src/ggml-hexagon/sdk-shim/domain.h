/*
 * Minimal stand-in for the Hexagon SDK <domain.h>.
 *
 * htp-drv.cpp references a table named supported_domains[] and the type
 * "domain". In the SDK both come from domain.h. Only htpdrv_get_domain() reads
 * it, and nothing in ggml-hexagon calls htpdrv_get_domain(), so the contents are
 * inert -- they exist to keep the API surface and the sizeof() expression valid.
 * The *live* domain ids come from FASTRPC_GET_DOMAINS or from the config.
 */
#pragma once

#include "AEEStdDef.h"

typedef struct domain {
    int  id;
    char name[64];
} domain;

static domain supported_domains[] = {
    { 0, "cdsp" },
    { 1, "cdsp" },
    { 2, "cdsp" },
    { 3, "cdsp" },
};
