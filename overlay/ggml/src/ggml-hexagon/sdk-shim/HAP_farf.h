/*
 * Minimal stand-in for the Hexagon SDK <HAP_farf.h>.
 *
 * On the DSP, FARF() is the lightweight tracing primitive and HAP_debug_v2() is
 * the runtime it forwards to. On the host, the generated qaic stub only touches
 * these on error paths, so routing them to stderr is enough.
 *
 * HAP_debug_v2 is *defined* here rather than left undefined: nothing in the
 * Android NDK exports it (on device it lives in the vendor's libcdsprpc, which
 * this build deliberately does not link against).
 */
#pragma once

#include <stdarg.h>
#include <stdio.h>

#ifndef RUNTIME_ERROR
#   define RUNTIME_ERROR 6
#endif

static inline int HAP_debug_v2(int level, const char * file, int line, const char * fmt, ...) {
    (void) level;
    (void) file;
    (void) line;
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    return 0;
}

#define FARF(level, ...)                    \
    do {                                    \
        fprintf(stderr, __VA_ARGS__);       \
    } while (0)
