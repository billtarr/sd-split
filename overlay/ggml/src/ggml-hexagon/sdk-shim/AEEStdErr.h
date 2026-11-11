/*
 * Minimal stand-in for the Hexagon SDK <AEEStdErr.h>.
 *
 * The numeric values matter: libcdsprpc.so returns these codes verbatim and
 * ggml-hexagon compares against AEE_EEXPIRED / AEE_EWOULDBLOCK to decide
 * whether a dspqueue_read() simply timed out (retry) or genuinely failed.
 *
 * In particular AEE_EEXPIRED is 12, *not* an offset-encoded value -- a wrong
 * guess here turns every "no response yet" into a hard abort. Confirmed on
 * device: dspqueue_read() returns 0x0000000c when the DSP has not answered
 * within the timeout.
 */
#pragma once

#include "AEEStdDef.h"

#define AEE_SUCCESS           0
#define AEE_EFAILED           1
#define AEE_ENOMEMORY         2
#define AEE_EUNSUPPORTED      3
#define AEE_EBADPARM          4
#define AEE_EBADCLASS         5
#define AEE_EBADMASK          6
#define AEE_EBADITEM          7
#define AEE_EINVALIDFORMAT    8
#define AEE_EWOULDBLOCK       9
#define AEE_ENOMORE           10
#define AEE_ECLASSNOTSUPPORT  11
#define AEE_EEXPIRED          12
#define AEE_ENOTFOUND         13
#define AEE_EINTERRUPTED      14
#define AEE_ENOTIMPLEMENTED   15
#define AEE_EINVALIDITEM      16
#define AEE_ENOTALLOWED       17
#define AEE_EBADHANDLE        18
#define AEE_EOUTOFHANDLES     19
#define AEE_EVERSIONNOTSUPPORT 20
#define AEE_EHANDLEINUSE      21
#define AEE_EBADSTATE         22
#define AEE_ETIMEOUT          23

/* Errors raised on the DSP side carry this bias (adsprpc: 0x80000400 + code). */
#define AEE_EOFFSET           0x80000400

/* Not present on this driver/SDK version. Only its low byte is ever compared. */
#define AEE_EUNSUPPORTEDAPI   (AEE_EOFFSET + 3)

/* Could not load the module / symbol (driver side). */
#define AEE_EUNABLETOLOAD     (AEE_EOFFSET + 2)

#ifndef AEE_SUCCESS_STR
#    define AEE_SUCCESS_STR "Success"
#endif
