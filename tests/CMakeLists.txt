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
function(mbr_add_test name)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${name}.cpp")
        return()
    endif()
    
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE test_common)
    add_test(NAME ${name} COMMAND ${name})
    
    # Copy DLLs on Windows
    mbr_copy_ggml_dlls(${name})
endfunction()

# Core tests (no external data required)
# Core tests (no external data required)
# mbr_add_test(test_audio) -> test_audio needs src/audio.cpp
add_executable(test_audio test_audio.cpp ../src/audio.cpp)
target_link_libraries(test_audio PRIVATE test_common)
target_include_directories(test_audio PRIVATE ../src ../third_party)
add_test(NAME test_audio COMMAND test_audio)
mbr_copy_ggml_dlls(test_audio)
mbr_add_test(test_component_stft)

# Component tests (require model + test data)
mbr_add_test(test_component_bandsplit)
mbr_add_test(test_component_layers)
mbr_add_test(test_component_mask)

# Integration tests
mbr_add_test(test_inference)
mbr_add_test(test_chunking_logic)
