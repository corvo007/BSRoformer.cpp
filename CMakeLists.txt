cmake_minimum_required(VERSION 3.17)
project(bs_roformer_cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#================================================
# Build Options
#================================================

option(GGML_CUDA "Enable CUDA backend" ON)
option(BSR_BUILD_TESTS "Build tests" OFF)
option(BSR_BUILD_CLI "Build CLI application" ON)
option(BSR_COPY_CUDA_DLLS "Copy CUDA runtime DLLs next to bs_roformer-cli on Windows" ON)

#================================================
# Dependencies - GGML (Flexible Resolution)
#================================================

# Strategy: Allow ggml to be shared across multiple projects
# 1. Check if ggml target already exists (e.g., from parent project like whisper.cpp)
# 2. If not, prefer ggml source via GGML_DIR/submodule/sibling directory
# 3. If not available, try to find ggml via CMAKE_PREFIX_PATH / package registry

if(NOT TARGET ggml)
    # Prefer ggml source directory if available. This avoids picking up broken/partial ggml
    # package configs from build directories or the user package registry.
    #
    # Priority 1: GGML_DIR variable (explicitly set by user or parent project)
    # Priority 2: Submodule in ggml/
    # Priority 3: Sibling directory ../ggml

    if(DEFINED GGML_DIR AND EXISTS "${GGML_DIR}/CMakeLists.txt")
        set(GGML_PATH "${GGML_DIR}")
        message(STATUS "Using GGML from GGML_DIR: ${GGML_PATH}")
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ggml/CMakeLists.txt")
        set(GGML_PATH "${CMAKE_CURRENT_SOURCE_DIR}/ggml")
        message(STATUS "Using GGML from submodule: ${GGML_PATH}")
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../ggml/CMakeLists.txt")
        set(GGML_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../ggml")
        message(STATUS "Using GGML from sibling directory: ${GGML_PATH}")
    else()
        # Fall back to finding an installed package (system-wide install or parent toolchain)
        find_package(ggml REQUIRED CONFIG)
        message(STATUS "Using GGML from installed package")
    endif()

    if(DEFINED GGML_PATH)
        # Force ggml to build as static libraries
        set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libraries" FORCE)

        # Add ggml as subdirectory
        add_subdirectory(${GGML_PATH} ggml EXCLUDE_FROM_ALL)
    endif()
else()
    message(STATUS "Using GGML target from parent project")
endif()

#================================================
# Core Library
#================================================

set(BSR_SOURCES
    src/model.cpp
    src/utils.cpp
    src/inference.cpp
)

add_library(bs_roformer STATIC ${BSR_SOURCES})

target_include_directories(bs_roformer PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(bs_roformer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(bs_roformer PUBLIC ggml)
if(GGML_CUDA AND TARGET ggml-cuda)
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
# CLI Application
#================================================

if(BSR_BUILD_CLI)
    # audio.cpp implements AudioFile utilities (using dr_wav)
    add_executable(bs-roformer-cli
        cli/main.cpp
        cli/process_utils.cpp
        src/audio.cpp
    )
    target_link_libraries(bs-roformer-cli PRIVATE bs_roformer)
    target_include_directories(bs-roformer-cli PRIVATE
        src
        third_party
    )
    
    if(MSVC)
        target_compile_options(bs-roformer-cli PRIVATE /W3 /utf-8)
    endif()

    if(UNIX AND NOT APPLE)
        # Allow colocated deployment on Linux:
        # - shared libs next to the executable ($ORIGIN)
        # - shared libs in a sibling lib directory ($ORIGIN/lib)
        set_target_properties(bs-roformer-cli PROPERTIES
            BUILD_RPATH "\$ORIGIN;\$ORIGIN/lib"
            INSTALL_RPATH "\$ORIGIN;\$ORIGIN/lib"
        )
    endif()

    if(WIN32 AND GGML_CUDA AND TARGET ggml-cuda AND BSR_COPY_CUDA_DLLS)
        if(NOT CUDAToolkit_BIN_DIR)
            find_package(CUDAToolkit REQUIRED)
        endif()

        if(CUDAToolkit_BIN_DIR)
            set(BSR_CUDA_RUNTIME_DLLS "")
            foreach(pattern
                    cudart64_*.dll
                    cublas64_*.dll
                    cublasLt64_*.dll)
                file(GLOB dlls "${CUDAToolkit_BIN_DIR}/${pattern}")
                list(APPEND BSR_CUDA_RUNTIME_DLLS ${dlls})
            endforeach()
            list(REMOVE_DUPLICATES BSR_CUDA_RUNTIME_DLLS)

            if(BSR_CUDA_RUNTIME_DLLS)
                foreach(dll ${BSR_CUDA_RUNTIME_DLLS})
                    add_custom_command(TARGET bs-roformer-cli POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                                "${dll}"
                                "$<TARGET_FILE_DIR:bs-roformer-cli>"
                        VERBATIM
                    )
                endforeach()
                message(STATUS "CUDA runtime DLL copy: ENABLED (${CUDAToolkit_BIN_DIR})")
            else()
                message(WARNING "BSR_COPY_CUDA_DLLS=ON but no CUDA runtime DLLs were found in ${CUDAToolkit_BIN_DIR}")
            endif()
        else()
            message(WARNING "BSR_COPY_CUDA_DLLS=ON but CUDAToolkit_BIN_DIR is not available")
        endif()
    endif()
endif()

#================================================
# Tests (Optional)
#================================================

if(BSR_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
    message(STATUS "Tests: ENABLED")
else()
    message(STATUS "Tests: DISABLED (use -DBSR_BUILD_TESTS=ON to enable)")
endif()
