target_sources(ninfer_engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/package.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/load/bindings.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/variant.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/tp2_comm.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/tp2_mailbox.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/tp2_wire8.cu)

target_include_directories(ninfer_engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/impl
  ${CMAKE_CURRENT_SOURCE_DIR}/export)

# TP2 collectives: an sm_70-capable NCCL (2.21 wheel build); the system NCCL has no Volta kernels.
# NCCL with sm_70 kernels. The stock NCCL 2.31 on Ubuntu 24.04 has no Volta kernels
# ("named symbol not found" at the first all-reduce); the pip wheel nvidia-nccl-cu12==2.21.5
# (or the 2.27.x wheel bundled with recent torch) works. Point this at the unpacked wheel:
#   -DNINFER_NCCL_ROOT=/path/to/nvidia/nccl   (contains include/nccl.h and lib/libnccl.so.2)
set(NINFER_NCCL_ROOT "$ENV{NINFER_NCCL_ROOT}" CACHE PATH "NCCL install prefix with sm_70 kernels")
if(NOT NINFER_NCCL_ROOT OR NOT EXISTS "${NINFER_NCCL_ROOT}/include/nccl.h")
  message(FATAL_ERROR "NINFER_NCCL_ROOT must point to an NCCL build with sm_70 kernels (see README, section 编译)")
endif()
target_include_directories(ninfer_engine PRIVATE ${NINFER_NCCL_ROOT}/include)
target_link_libraries(ninfer_engine PUBLIC ${NINFER_NCCL_ROOT}/lib/libnccl.so.2)
set(CMAKE_BUILD_RPATH "${NINFER_NCCL_ROOT}/lib")
