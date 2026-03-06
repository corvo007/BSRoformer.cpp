#include "test_common.h"
#include "../cli/process_utils.h"

#include <string>

int main(int argc, char* argv[]) {
    if (argc >= 2 && std::string(argv[1]) == "--child-process-smoke") {
        TEST_ASSERT(argc == 4, "Expected 2 forwarded child arguments");
        TEST_ASSERT(std::string(argv[2]) == "arg with spaces", "First forwarded arg mismatch");
        TEST_ASSERT(std::string(argv[3]) == "special=123", "Second forwarded arg mismatch");
        return 23;
    }

    std::cout << "Test: CLI process helper" << std::endl;

    const auto self_path = cli_process::GetSelfExecutablePath(argv[0]);
    TEST_ASSERT(!self_path.empty(), "Self executable path should not be empty");

    const int rc = cli_process::SpawnChildAndWait(
        self_path,
        {"--child-process-smoke", "arg with spaces", "special=123"}
    );

    TEST_ASSERT(rc == 23, "Child exit code should be propagated");

    LOG_PASS();
    return 0;
}
