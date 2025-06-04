cmake_minimum_required(VERSION 3.18)
project(vllm_kernels LANGUAGES CXX CUDA C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find PyTorch to link against and get include directories
find_package(Torch REQUIRED)
# PyTorch new versions does not expose an interface with pybind11,
# instead it uses its own c10::pybind11_object.
# For older versions, we still need to find pybind11.
if(NOT TARGET Torch::pybind11_headers)
  find_package(pybind11 REQUIRED)
endif()

# Set CMAKE_CUDA_ARCHITECTURES if not set by user or environment
# VLLM_TARGET_DEVICE is passed from setup.py and used to conditionally compile code.
if(VLLM_TARGET_DEVICE STREQUAL "cuda")
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND DEFINED ENV{TORCH_CUDA_ARCH_LIST})
        string(REPLACE ";" " " TORCH_CUDA_ARCH_LIST_SPACES "$ENV{TORCH_CUDA_ARCH_LIST}")
        set(CMAKE_CUDA_ARCHITECTURES_TEMP "")
        foreach(ARCH ${TORCH_CUDA_ARCH_LIST_SPACES})
            string(REPLACE "." "" ARCH_NO_DOT ${ARCH})
            if(ARCH_NO_DOT MATCHES "([0-9]+)\\+PTX")
                list(APPEND CMAKE_CUDA_ARCHITECTURES_TEMP "${CMAKE_MATCH_1}-virtual")
            else()
                list(APPEND CMAKE_CUDA_ARCHITECTURES_TEMP "${ARCH_NO_DOT}-real")
            endif()
        endforeach()
        string(REPLACE " " ";" CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES_TEMP}")
        unset(CMAKE_CUDA_ARCHITECTURES_TEMP)
    elseif(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86") # Sensible default if not specified
    endif()
    message(STATUS "vllm-kernels: Building for CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # Add other CUDA specific flags if necessary, e.g. from Torch_CUDA_FLAGS
    # list(APPEND CMAKE_CUDA_FLAGS "${Torch_CUDA_FLAGS}")
endif()


include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc"  # Sources are in vllm-kernels/csrc
    "${Torch_INCLUDE_DIRS}"
)
if(TARGET pybind11::headers)
  include_directories("${pybind11_INCLUDE_DIRS}")
endif()

# Global compile options (can be overridden per target)
if(MSVC)
    add_compile_options(/W4 /MP /wd4251 /wd4275 /wd4819 /wd4244 /wd4267)
    add_compile_definitions(NOMINMAX)
else()
    add_compile_options(-Wall -Wextra -Wno-unused-parameter -fPIC)
endif()


# Function to add a Python extension module
# NAME: The short leaf name of the module (e.g., _C, _vllm_fa2_C)
# SOURCES: List of source files relative to ${CMAKE_CURRENT_SOURCE_DIR}
function(add_vllm_kernel_module NAME)
    set(options "")
    set(oneValueArgs "") # No single value args apart from NAME
    set(multiValueArgs SOURCES LIBS DEFINES INCLUDE_DIRS CUDA_SOURCES)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TARGET_NAME ${NAME})
    set(ALL_SOURCES ${ARG_SOURCES})

    if(ARG_CUDA_SOURCES AND (VLLM_TARGET_DEVICE STREQUAL "cuda" OR VLLM_TARGET_DEVICE STREQUAL "rocm"))
        list(APPEND ALL_SOURCES ${ARG_CUDA_SOURCES})
    endif()

    add_library(${TARGET_NAME} SHARED ${ALL_SOURCES})
    target_compile_features(${TARGET_NAME} PRIVATE cxx_std_17)

    target_link_libraries(${TARGET_NAME} PRIVATE Torch::torch Torch::torch_cpu Torch::c10)
    if(VLLM_TARGET_DEVICE STREQUAL "cuda" OR VLLM_TARGET_DEVICE STREQUAL "rocm")
      target_link_libraries(${TARGET_NAME} PRIVATE Torch::torch_cuda Torch::c10_cuda)
    endif()

    if(TARGET pybind11::pybind11)
        target_link_libraries(${TARGET_NAME} PRIVATE pybind11::pybind11)
    elseif(TARGET pybind11::embed) # Some setups might expose embed
         target_link_libraries(${TARGET_NAME} PRIVATE pybind11::embed)
    endif()
    
    if(ARG_LIBS)
        target_link_libraries(${TARGET_NAME} PRIVATE ${ARG_LIBS})
    endif()

    if(ARG_DEFINES)
        target_compile_definitions(${TARGET_NAME} PRIVATE ${ARG_DEFINES})
    endif()

    if(ARG_INCLUDE_DIRS)
        target_include_directories(${TARGET_NAME} PRIVATE ${ARG_INCLUDE_DIRS})
    endif()
    
    set_target_properties(${TARGET_NAME} PROPERTIES PREFIX "" LIBRARY_OUTPUT_NAME "${NAME}")
    # This definition is crucial for PyTorch C++ extensions
    target_compile_definitions(${TARGET_NAME} PRIVATE TORCH_EXTENSION_NAME=${NAME})

    if(CMAKE_CUDA_COMPILER AND ARG_CUDA_SOURCES)
      target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
          ${TORCH_CUDA_FLAGS} # Use flags from PyTorch
          # Add any vLLM specific CUDA flags here if needed
          # Example: -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ --expt-relaxed-constexpr
      >)
    endif()

endfunction()


# Build _C module (General Kernels)
set(COMMON_KERNEL_SOURCES
    "csrc/pybind.cpp" # Main pybind file for _C
)
set(COMMON_KERNEL_CUDA_SOURCES
    "csrc/attention/attention_kernels.cu"
    "csrc/cache_kernels.cu"
    "csrc/cuda_utils.cu"
    "csrc/activation_kernels.cu"
    "csrc/pos_encoding_kernels.cu"
    "csrc/layernorm_kernels.cu"
    "csrc/quantization/awq/gemm_kernels.cu"
    "csrc/quantization/squeezellm/quant_kernels.cu"
    "csrc/quantization/gptq/gptq_kernel.cu"
    "csrc/quantization/marlin/marlin_cuda_kernel.cu"
    "csrc/quantization/fp8/fp8_ops.cu"
    # Add other general .cu files that compile into _C.so
)
add_vllm_kernel_module(_C SOURCES ${COMMON_KERNEL_SOURCES} CUDA_SOURCES ${COMMON_KERNEL_CUDA_SOURCES})


# Build _moe_C module (MoE Kernels)
if(VLLM_TARGET_DEVICE STREQUAL "cuda" OR VLLM_TARGET_DEVICE STREQUAL "rocm")
    set(MOE_KERNEL_SOURCES "csrc/moe/moe_ops.cpp") # Pybind for MoE
    set(MOE_KERNEL_CUDA_SOURCES
        "csrc/moe/moe_kernels.cu"
        "csrc/moe/marlin_moe.cu"
    )
    add_vllm_kernel_module(_moe_C SOURCES ${MOE_KERNEL_SOURCES} CUDA_SOURCES ${MOE_KERNEL_CUDA_SOURCES})
endif()

# Build _rocm_C module (ROCm Specific Kernels)
if(VLLM_TARGET_DEVICE STREQUAL "rocm")
    set(ROCM_KERNEL_SOURCES "csrc/rocm_ops.cpp") # Pybind for ROCm specific ops
    set(ROCM_KERNEL_CUDA_SOURCES # hip sources
        "csrc/rocm/attention_kernels.cu"
        "csrc/rocm/pos_encoding_kernels.cu"
        "csrc/rocm/misc_kernels.cu"
        "csrc/rocm/cache_kernels.cu"
        "csrc/rocm/activation_kernels.cu"
        "csrc/rocm/layernorm_kernels.cu"
        "csrc/rocm/sampler_kernels.cu"
    )
    add_vllm_kernel_module(_rocm_C SOURCES ${ROCM_KERNEL_SOURCES} CUDA_SOURCES ${ROCM_KERNEL_CUDA_SOURCES})
endif()

# Build cumem_allocator module
if(VLLM_TARGET_DEVICE STREQUAL "cuda") # Typically CUDA specific
    add_vllm_kernel_module(cumem_allocator SOURCES "csrc/cumem_allocator.cpp")
endif()

# Build _flashmla_C module
if(VLLM_TARGET_DEVICE STREQUAL "cuda")
    set(FLASHMLA_SOURCES "csrc/flashmla/flashmla_ops.cpp") # Pybind for FlashMLA
    set(FLASHMLA_CUDA_SOURCES "csrc/flashmla/flashmla_kernels.cu")
    add_vllm_kernel_module(_flashmla_C SOURCES ${FLASHMLA_SOURCES} CUDA_SOURCES ${FLASHMLA_CUDA_SOURCES})
endif()

# FlashAttention Kernels (_vllm_fa2_C, _vllm_fa3_C)
if(VLLM_TARGET_DEVICE STREQUAL "cuda")
    set(FLASH_ATTN_INCLUDE_DIRS
        "${CMAKE_CURRENT_SOURCE_DIR}/csrc/flash_attn/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/csrc/flash_attn/cutlass_kernels/include"
    )
    set(FLASH_ATTN_COMMON_CUDA_SOURCES
        "csrc/flash_attn/src/flash_fwd_hdim32_fp16_sm80.cu"
        "csrc/flash_attn/src/flash_fwd_hdim32_bf16_sm80.cu"
        "csrc/flash_attn/src/flash_fwd_hdim64_fp16_sm80.cu"
        "csrc/flash_attn/src/flash_fwd_hdim64_bf16_sm80.cu"
        "csrc/flash_attn/src/flash_fwd_hdim96_fp16_sm80.cu"
        "csrc/flash_attn/src/flash_fwd_hdim96_bf16_sm80.cu"
        "csrc/flash_attn/src/flash_fwd_hdim128_fp16_sm80.cu"
        "csrc/flash_attn/src/flash_fwd_hdim128_bf16_sm80.cu"
        "csrc/flash_attn/src/flash_fwd_hdim160_fp16_sm80.cu"
        "csrc/flash_attn/src/flash_fwd_hdim160_bf16_sm80.cu"
        "csrc/flash_attn/src/flash_fwd_hdim192_fp16_sm80.cu"
        "csrc/flash_attn/src/flash_fwd_hdim192_bf16_sm80.cu"
        "csrc/flash_attn/src/flash_fwd_hdim224_fp16_sm80.cu"
        "csrc/flash_attn/src/flash_fwd_hdim224_bf16_sm80.cu"
        "csrc/flash_attn/src/flash_fwd_hdim256_fp16_sm80.cu"
        "csrc/flash_attn/src/flash_fwd_hdim256_bf16_sm80.cu"
        "csrc/flash_attn/src/flash_bwd_hdim32_fp16_sm80.cu"
        "csrc/flash_attn/src/flash_bwd_hdim32_bf16_sm80.cu"
        "csrc/flash_attn/src/flash_bwd_hdim64_fp16_sm80.cu"
        "csrc/flash_attn/src/flash_bwd_hdim64_bf16_sm80.cu"
        "csrc/flash_attn/src/flash_bwd_hdim96_fp16_sm80.cu"
        "csrc/flash_attn/src/flash_bwd_hdim96_bf16_sm80.cu"
        "csrc/flash_attn/src/flash_bwd_hdim128_fp16_sm80.cu"
        "csrc/flash_attn/src/flash_bwd_hdim128_bf16_sm80.cu"
        "csrc/flash_attn/src/flash_bwd_hdim160_fp16_sm80.cu"
        "csrc/flash_attn/src/flash_bwd_hdim160_bf16_sm80.cu"
        "csrc/flash_attn/src/flash_bwd_hdim192_fp16_sm80.cu"
        "csrc/flash_attn/src/flash_bwd_hdim192_bf16_sm80.cu"
        "csrc/flash_attn/src/flash_bwd_hdim224_fp16_sm80.cu"
        "csrc/flash_attn/src/flash_bwd_hdim224_bf16_sm80.cu"
        "csrc/flash_attn/src/flash_bwd_hdim256_fp16_sm80.cu"
        "csrc/flash_attn/src/flash_bwd_hdim256_bf16_sm80.cu"
        "csrc/flash_attn/src/flash_fwd_launch_template.cu"
        "csrc/flash_attn/src/flash_bwd_launch_template.cu"
        "csrc/flash_attn/cutlass_kernels/fpA_intB_gemm_wrapper.cu"
    )
    set(FLASH_ATTN_COMMON_CPP_SOURCES "csrc/flash_attn/src/static_switch.cpp")

    # Build _vllm_fa2_C module
    set(FA2_SOURCES "csrc/flash_attn/fa2_api.cpp" ${FLASH_ATTN_COMMON_CPP_SOURCES})
    add_vllm_kernel_module(_vllm_fa2_C SOURCES ${FA2_SOURCES} CUDA_SOURCES ${FLASH_ATTN_COMMON_CUDA_SOURCES} INCLUDE_DIRS ${FLASH_ATTN_INCLUDE_DIRS})

    # Build _vllm_fa3_C module
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.3" OR CMAKE_CUDA_COMPILER_ID STREQUAL "Clang") # Clang may report different versions
        set(FA3_SOURCES "csrc/flash_attn/fa3_api.cpp" ${FLASH_ATTN_COMMON_CPP_SOURCES}) # Assuming fa3_api.cpp for FA3 bindings
        add_vllm_kernel_module(_vllm_fa3_C SOURCES ${FA3_SOURCES} CUDA_SOURCES ${FLASH_ATTN_COMMON_CUDA_SOURCES} INCLUDE_DIRS ${FLASH_ATTN_INCLUDE_DIRS})
    endif()
endif() 