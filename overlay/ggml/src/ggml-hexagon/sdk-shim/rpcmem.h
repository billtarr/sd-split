/*
 * Minimal stand-in for the Hexagon SDK <rpcmem.h>.
 *
 * The prototypes below are satisfied by htp-drv.cpp, which implements them as
 * thin wrappers over dlsym()'d pointers into libcdsprpc.so. Signatures must stay
 * in sync with that file.
 */
#pragma once

#include "AEEStdDef.h"

/* System heap, the only one used by ggml-hexagon. */
#define RPCMEM_HEAP_ID_SYSTEM 25

/* Default allocation flag (cached, no special IOMMU hint). */
#define RPCMEM_DEFAULT_FLAGS 1

/* Deprecated alias kept for source compatibility. */
#define RPCMEM_FREE 0

#ifdef __cplusplus
extern "C" {
#endif

void * rpcmem_alloc (int heapid, uint32_t flags, int    size);
void * rpcmem_alloc2(int heapid, uint32_t flags, size_t size);
void   rpcmem_free  (void * po);
int    rpcmem_to_fd (void * po);

#ifdef __cplusplus
}
#endif
