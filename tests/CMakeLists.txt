# tests/CMakeLists.txt
# Test suite for BSRoformer

#================================================
# Test Infrastructure
#================================================

add_library(test_common STATIC test_common.cpp)
target_link_libraries(test_common PUBLIC bs_roformer)
target_include_directories(test_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

#================================================
# Test Registration
#================================================

# Helper: Add a test with common configuration
function(bsr_add_test name)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${name}.cpp")
        return()
    endif()
    
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE test_common)
    add_test(NAME ${name} COMMAND ${name})
    
    # Copy DLLs on Windows
    bsr_copy_ggml_runtime_dlls(${name})
endfunction()

# Core tests (no external data required)
# bsr_add_test(test_audio) -> test_audio needs src/audio.cpp
add_executable(test_audio test_audio.cpp ../src/audio.cpp)
target_link_libraries(test_audio PRIVATE test_common)
target_include_directories(test_audio PRIVATE ../src ../third_party)
add_test(NAME test_audio COMMAND test_audio)
bsr_copy_ggml_runtime_dlls(test_audio)
bsr_add_test(test_component_stft)

# Component tests (require model + test data)
bsr_add_test(test_component_bandsplit)
bsr_add_test(test_component_layers)
bsr_add_test(test_component_mask)

# Integration tests
bsr_add_test(test_inference)
bsr_add_test(test_chunking_logic)
bsr_add_test(test_stft_consistency)
bsr_add_test(test_cancel_callback)
