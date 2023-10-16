cmake_minimum_required(VERSION 3.5)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70)
endif()
project(SpMV CUDA)

add_executable(spmv.x main_csr.cu)