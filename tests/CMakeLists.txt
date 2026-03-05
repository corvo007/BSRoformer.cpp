# tests/CMakeLists.txt
# Test suite for BSRoformer

#================================================
# Test Infrastructure
#================================================

add_library(test_common STATIC test_common.cpp)
target_link_libraries(test_common PUBLIC bs_roformer)
target_include_directories(test_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
if(MSVC)
    target_compile_options(test_common PRIVATE /W3 /utf-8)
endif()

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
    if(MSVC)
        target_compile_options(${name} PRIVATE /W3 /utf-8)
    endif()
    add_test(NAME ${name} COMMAND ${name})
endfunction()

# Core tests (no external data required)
# bsr_add_test(test_audio) -> test_audio needs src/audio.cpp
add_executable(test_audio test_audio.cpp ../src/audio.cpp)
target_link_libraries(test_audio PRIVATE test_common)
target_include_directories(test_audio PRIVATE ../src ../third_party)
if(MSVC)
    target_compile_options(test_audio PRIVATE /W3 /utf-8)
endif()
add_test(NAME test_audio COMMAND test_audio)
bsr_add_test(test_component_stft)

# Needs AudioFile::Load (src/audio.cpp)
add_executable(test_inference_offset_stability test_inference_offset_stability.cpp ../src/audio.cpp)
target_link_libraries(test_inference_offset_stability PRIVATE test_common)
target_include_directories(test_inference_offset_stability PRIVATE ../src ../third_party)
if(MSVC)
    target_compile_options(test_inference_offset_stability PRIVATE /W3 /utf-8)
endif()
add_test(NAME test_inference_offset_stability COMMAND test_inference_offset_stability)

# Component tests (require model + test data)
bsr_add_test(test_component_bandsplit)
bsr_add_test(test_component_layers)
bsr_add_test(test_component_mask)

# Integration tests
bsr_add_test(test_inference)
bsr_add_test(test_chunking_logic)
bsr_add_test(test_stft_consistency)
bsr_add_test(test_cancel_callback)
bsr_add_test(test_streaming_overlap_add)
bsr_add_test(test_inference_repeatability)

# Some tests depend on external model/test-data. If the required files are missing,
# they return code 77 and CTest will report them as "Skipped" instead of "Failed".
set_tests_properties(test_component_bandsplit PROPERTIES SKIP_RETURN_CODE 77)
set_tests_properties(test_component_layers PROPERTIES SKIP_RETURN_CODE 77)
set_tests_properties(test_component_mask PROPERTIES SKIP_RETURN_CODE 77)
set_tests_properties(test_inference PROPERTIES SKIP_RETURN_CODE 77)
set_tests_properties(test_chunking_logic PROPERTIES SKIP_RETURN_CODE 77)
set_tests_properties(test_stft_consistency PROPERTIES SKIP_RETURN_CODE 77)
set_tests_properties(test_inference_repeatability PROPERTIES SKIP_RETURN_CODE 77)
set_tests_properties(test_inference_offset_stability PROPERTIES SKIP_RETURN_CODE 77)
