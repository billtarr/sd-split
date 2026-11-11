/*
 * Minimal stand-in for the Hexagon SDK <dspqueue.h>.
 *
 * dspqueue_buffer is the *ABI* shared with libcdsprpc.so: it is handed to
 * dspqueue_write()/dspqueue_read() as the descriptor of every shared buffer, so
 * the field order and sizes below must match the library exactly.
 *
 * Layout was recovered by disassembling the prebuilt libggml-hexagon.so
 * (ggml_hexagon_opqueue::push) and libcdsprpc.so (dspqueue_write_noblock):
 *
 *   str  x27, [dbuf, #0x10]       <- void* ptr
 *   stp  w11, w13, [dbuf, #0x8]   <- uint32 offset, uint32 flags
 *   stp  w10, w8,  [dbuf]         <- int fd, uint32 size
 *
 * i.e. fd first and the host pointer last -- NOT the other way round. Getting
 * this wrong makes the library read the low half of `ptr` as a file descriptor
 * and fail with "Buffer FD ... in queue message not mapped to domain".
 *
 * The dspqueue_* functions themselves are thin wrappers over dlsym()'d
 * pointers, implemented in htp-drv.cpp.
 */
#pragma once

#include "AEEStdErr.h"
#include "AEEStdDef.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dspqueue * dspqueue_t;

struct dspqueue_buffer {
    int      fd;      /* dmabuf fd */
    uint32_t size;    /* valid bytes */
    uint32_t offset;  /* byte offset of the payload inside the buffer */
    uint32_t flags;   /* DSPQUEUE_BUFFER_FLAG_* */
    void *   ptr;     /* host virtual address of the payload */
};

/* Together these make up the 0x90 the reference implementation writes. */
#define DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT 0x10
#define DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER         0x80

/* Block forever. htp-drv.h overrides DSPQUEUE_TIMEOUT with a real budget. */
#ifndef DSPQUEUE_TIMEOUT
#    define DSPQUEUE_TIMEOUT 0xFFFFFFFFu
#endif
/* Do not block. */
#ifndef DSPQUEUE_TIMEOUT_NONE
#    define DSPQUEUE_TIMEOUT_NONE 0x0u
#endif

typedef void (*dspqueue_callback_t)(int error, void * context);

AEEResult dspqueue_create(int                 domain,
                          uint32_t            flags,
                          uint32_t            req_queue_size,
                          uint32_t            resp_queue_size,
                          dspqueue_callback_t packet_callback,
                          dspqueue_callback_t error_callback,
                          void *              callback_context,
                          dspqueue_t *        queue);

AEEResult dspqueue_close(dspqueue_t queue);

AEEResult dspqueue_export(dspqueue_t queue, uint64_t * queue_id);

AEEResult dspqueue_write(dspqueue_t               queue,
                         uint32_t                 flags,
                         uint32_t                 num_buffers,
                         struct dspqueue_buffer * buffers,
                         uint32_t                 message_length,
                         const uint8_t *          message,
                         uint32_t                 timeout_us);

AEEResult dspqueue_read(dspqueue_t               queue,
                        uint32_t *               flags,
                        uint32_t                 max_buffers,
                        uint32_t *               num_buffers,
                        struct dspqueue_buffer * buffers,
                        uint32_t                 max_message_length,
                        uint32_t *               message_length,
                        uint8_t *                message,
                        uint32_t                 timeout_us);

AEEResult dspqueue_read_noblock(dspqueue_t               queue,
                                uint32_t *               flags,
                                uint32_t                 max_buffers,
                                uint32_t *               num_buffers,
                                struct dspqueue_buffer * buffers,
                                uint32_t                 max_message_length,
                                uint32_t *               message_length,
                                uint8_t *                message);

#ifdef __cplusplus
}
#endif
