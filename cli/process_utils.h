#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace cli_process {

std::filesystem::path GetSelfExecutablePath(const char* argv0);

int SpawnChildAndWait(const std::filesystem::path& exe_path, const std::vector<std::string>& args);

} // namespace cli_process
