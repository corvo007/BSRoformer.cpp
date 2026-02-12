cmake_minimum_required(VERSION 3.17)
project(bs_roformer_cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#================================================
# Build Options
#================================================

option(GGML_CUDA "Enable CUDA backend" ON)
option(MBR_BUILD_TESTS "Build tests" OFF)
option(MBR_BUILD_CLI "Build CLI application" ON)

#================================================
# Dependencies - GGML (Flexible Resolution)
#================================================

# Strategy: Allow ggml to be shared across multiple projects
# 1. Check if ggml target already exists (e.g., from parent project like whisper.cpp)
# 2. If not, try to find ggml via CMAKE_PREFIX_PATH or GGML_DIR
# 3. If not found, use local ggml (submodule or sibling directory)

if(NOT TARGET ggml)
    # Try to find ggml package first (for system-wide or parent project installation)
    find_package(ggml QUIET CONFIG)
    
    if(NOT ggml_FOUND)
        # ggml not found as package, look for source directory
        # Priority 1: GGML_DIR variable (explicitly set by user or parent project)
        # Priority 2: Submodule in ggml/
        # Priority 3: Sibling directory ../ggml
        
        if(DEFINED GGML_DIR)
            set(GGML_PATH "${GGML_DIR}")
            message(STATUS "Using GGML from GGML_DIR: ${GGML_PATH}")
        elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ggml/CMakeLists.txt")
            set(GGML_PATH "${CMAKE_CURRENT_SOURCE_DIR}/ggml")
            message(STATUS "Using GGML from submodule: ${GGML_PATH}")
        elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../ggml/CMakeLists.txt")
            set(GGML_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../ggml")
            message(STATUS "Using GGML from sibling directory: ${GGML_PATH}")
        else()
            message(FATAL_ERROR 
                "ggml not found. Please either:\n"
                "  1. Add ggml as submodule: git submodule add https://github.com/ggerganov/ggml.git\n"
                "  2. Clone ggml to parent directory: cd .. && git clone https://github.com/ggerganov/ggml.git\n"
                "  3. Set GGML_DIR to point to ggml source: cmake -DGGML_DIR=/path/to/ggml\n"
                "  4. Let parent project provide ggml target"
            )
        endif()
        
        # Add ggml as subdirectory
        add_subdirectory(${GGML_PATH} ggml EXCLUDE_FROM_ALL)
    else()
        message(STATUS "Using GGML from installed package")
    endif()
else()
    message(STATUS "Using GGML target from parent project")
endif()

#================================================
# Core Library
#================================================

set(MBR_SOURCES
    src/model.cpp
    src/utils.cpp
    src/inference.cpp
)

add_library(bs_roformer STATIC ${MBR_SOURCES})

target_include_directories(bs_roformer PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(bs_roformer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(bs_roformer PUBLIC ggml)
if(GGML_CUDA AND TARGET ggml-cuda)
    target_link_libraries(bs_roformer PUBLIC ggml-cuda)
    
    # Fix for CI: Link against CUDA stubs if the driver is not present
    # This prevents errors like "libcuda.so.1 needed by ... not found" during linking
    find_package(CUDAToolkit REQUIRED)
    if(TARGET CUDA::cuda_driver)
        target_link_libraries(bs_roformer PUBLIC CUDA::cuda_driver)
        message(STATUS "Linked against CUDA driver stubs")
    endif()
endif()

# Compiler options
if(MSVC)
    target_compile_options(bs_roformer PRIVATE /W3 /utf-8)
else()
    target_compile_options(bs_roformer PRIVATE -Wall -Wextra)
endif()

# OpenMP support
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(bs_roformer PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(bs_roformer PUBLIC USE_OPENMP)
    message(STATUS "OpenMP: ENABLED")
else()
    message(STATUS "OpenMP: NOT FOUND")
endif()

#================================================
# DLL Copy Helper (Windows)
#================================================

function(mbr_copy_ggml_dlls target_name)
    if(NOT WIN32)
        return()
    endif()
    
    set(GGML_DLL_TARGETS ggml ggml-base ggml-cpu ggml-cuda ggml-vulkan)
    
    foreach(dll_target ${GGML_DLL_TARGETS})
        if(TARGET ${dll_target})
            add_custom_command(TARGET ${target_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_FILE:${dll_target}>
                    $<TARGET_FILE_DIR:${target_name}>
                COMMENT "Copying ${dll_target}.dll for ${target_name}"
            )
        endif()
    endforeach()
endfunction()

#================================================
# CLI Application
#================================================

if(MBR_BUILD_CLI)
    # audio.cpp implements AudioFile utilities (using dr_wav)
    add_executable(bs_roformer-cli 
        cli/main.cpp 
        src/audio.cpp
    )
    target_link_libraries(bs_roformer-cli PRIVATE bs_roformer)
    target_include_directories(bs_roformer-cli PRIVATE 
        src 
        third_party
    )
    
    if(MSVC)
        target_compile_options(bs_roformer-cli PRIVATE /W3 /utf-8)
    endif()
    
    mbr_copy_ggml_dlls(bs_roformer-cli)
endif()

#================================================
# Tests (Optional)
#================================================

if(MBR_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
    message(STATUS "Tests: ENABLED")
else()
    message(STATUS "Tests: DISABLED (use -DMBR_BUILD_TESTS=ON to enable)")
endif()
