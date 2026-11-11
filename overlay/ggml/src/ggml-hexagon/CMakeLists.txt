# ggml-hexagon — Hexagon (CDSP/HTP) backend, ported from llama.cpp into
# stable-diffusion.cpp's leejet/ggml fork.
#
# Differences from the upstream CMakeLists:
#   * No Qualcomm Hexagon SDK required. The SDK is account-gated and unavailable
#     here; every fastrpc / rpcmem / dspqueue entry point is resolved at runtime
#     with dlopen + dlsym (htp-drv.cpp), and the IDL stub is checked in
#     (htp_iface.h + htp_iface_stub.c, generated once by the SDK's qaic tool).
#     Types come from sdk-shim/.
#   * No DSP skel build. The applet (libggml-htp-<v>.so) is a device-side
#     artifact that is independent of the ggml version; we reuse the prebuilt
#     one and ship it in jniLibs / ADSP_LIBRARY_PATH.
#
# The only version-sensitive piece is the host side, which is adapted from
# ggml 0.29 (llama.cpp) to 0.27 (leejet fork).

set(TARGET_NAME ggml-hexagon)

ggml_add_backend_library(${TARGET_NAME}
    ggml-hexagon.cpp
    htp-common.h
    htp-kparams.cpp
    htp-kparams.h
    htp-opnode.cpp
    htp-opnode.h
    htp-fusion.cpp
    htp-fusion.h
    htp-repack.cpp
    htp-repack.h
    htp-drv.cpp
    htp-drv.h
    libdl.h
    htp_iface.h
    htp_iface_stub.c
    ../../include/ggml-hexagon.h)

target_include_directories(${TARGET_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/htp
    ${CMAKE_CURRENT_SOURCE_DIR}/sdk-shim
    ${CMAKE_CURRENT_BINARY_DIR})

if (CMAKE_SYSTEM_NAME MATCHES "Android")
    target_link_libraries(${TARGET_NAME} PRIVATE log dl)
elseif (CMAKE_SYSTEM_NAME MATCHES "Linux")
    target_link_libraries(${TARGET_NAME} PRIVATE dl)
endif()

# htp_iface_stub.c is C, the rest is C++. Keep the stub out of the C++ warning
# regime -- it is generated code.
set_source_files_properties(htp_iface_stub.c PROPERTIES COMPILE_OPTIONS "-w")

# Note: ggml_add_backend_library() already installs the target. There is no skel
# to install because the applet is a prebuilt device-side artifact.
