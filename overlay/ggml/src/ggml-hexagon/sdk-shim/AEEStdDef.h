/*
 * Minimal stand-in for the Hexagon SDK <AEEStdDef.h>.
 *
 * Why this file exists: the Qualcomm Hexagon SDK is not installable on this
 * machine (account-gated, and this build does not use the Windows toolchain).
 * The ggml-hexagon host side only needs the *types* from the SDK headers --
 * every fastrpc / rpcmem / dspqueue entry point is resolved at runtime with
 * dlopen + dlsym (see htp-drv.cpp). So a type-only shim is sufficient.
 *
 * Keep the layouts here byte-identical to the SDK, because these structs are
 * the ABI between this process and libcdsprpc.so.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

typedef int AEEResult;

#ifndef AEEAPI
#   define AEEAPI
#endif
#ifndef AEEAPIENTRY
#   define AEEAPIENTRY
#endif
#ifndef __QAIC_API
#   define __QAIC_API(f) f
#endif

typedef uint8_t  boolean;
typedef uint8_t  uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;
/*
 * Note: "unsigned long long", not uint64_t. On LP64 those are different types
 * (long vs long long), and the generated qaic header declares htp_iface_hwinfo
 * with uint64*, which is why the call site passes unsigned long long*.
 */
typedef unsigned long long uint64;
typedef signed char    int8;
typedef short          int16;
typedef int            int32;
typedef long long      int64;
