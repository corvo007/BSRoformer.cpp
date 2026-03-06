#include "process_utils.h"

#include <cstdlib>
#include <limits>
#include <cstring>
#include <stdexcept>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <process.h>
#else
#include <cerrno>
#include <spawn.h>
#include <sys/wait.h>
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

extern char** environ;
#endif

namespace cli_process {

#ifdef _WIN32
static std::wstring Utf8ToWide(const std::string& s) {
    if (s.empty()) return {};

    int needed = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    if (needed <= 0) {
        needed = MultiByteToWideChar(CP_ACP, 0, s.c_str(), -1, nullptr, 0);
        if (needed <= 0) return {};

        std::wstring out(static_cast<size_t>(needed) - 1, L'\0');
        MultiByteToWideChar(CP_ACP, 0, s.c_str(), -1, out.data(), needed);
        return out;
    }

    std::wstring out(static_cast<size_t>(needed) - 1, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, out.data(), needed);
    return out;
}

static std::wstring QuoteWindowsArg(const std::wstring& arg) {
    if (arg.empty()) {
        return L"\"\"";
    }

    const bool needs_quotes = arg.find_first_of(L" \t\n\v\"") != std::wstring::npos;
    if (!needs_quotes) {
        return arg;
    }

    std::wstring quoted;
    quoted.push_back(L'"');

    size_t backslash_count = 0;
    for (const wchar_t ch : arg) {
        if (ch == L'\\') {
            ++backslash_count;
            continue;
        }

        if (ch == L'"') {
            quoted.append(backslash_count * 2 + 1, L'\\');
            quoted.push_back(L'"');
            backslash_count = 0;
            continue;
        }

        quoted.append(backslash_count, L'\\');
        backslash_count = 0;
        quoted.push_back(ch);
    }

    quoted.append(backslash_count * 2, L'\\');
    quoted.push_back(L'"');
    return quoted;
}
#endif

std::filesystem::path GetSelfExecutablePath(const char* argv0) {
#ifdef _WIN32
    std::wstring buf;
    buf.resize(32768);
    DWORD len = GetModuleFileNameW(nullptr, buf.data(), static_cast<DWORD>(buf.size()));
    if (len > 0 && len < buf.size()) {
        buf.resize(len);
        return std::filesystem::path(buf);
    }
#elif defined(__APPLE__)
    uint32_t size = 1024;
    std::vector<char> buf(size);
    if (_NSGetExecutablePath(buf.data(), &size) != 0) {
        buf.resize(size);
        if (_NSGetExecutablePath(buf.data(), &size) != 0) {
            return std::filesystem::path(argv0 ? argv0 : "");
        }
    }
    return std::filesystem::path(buf.data());
#elif defined(__linux__)
    std::vector<char> buf(1024, '\0');
    while (true) {
        const ssize_t len = readlink("/proc/self/exe", buf.data(), buf.size() - 1);
        if (len < 0) {
            break;
        }
        if (static_cast<size_t>(len) < buf.size() - 1) {
            buf[static_cast<size_t>(len)] = '\0';
            return std::filesystem::path(buf.data());
        }
        buf.resize(buf.size() * 2, '\0');
    }
#endif
    return std::filesystem::path(argv0 ? argv0 : "");
}

#ifdef _WIN32
int SpawnChildAndWait(const std::filesystem::path& exe_path, const std::vector<std::string>& args) {
    const std::wstring exe = exe_path.wstring();
    if (exe.empty()) {
        throw std::runtime_error("Failed to spawn child process: executable path is empty");
    }

    std::wstring command_line = QuoteWindowsArg(exe);
    for (const auto& a : args) {
        command_line.push_back(L' ');
        command_line += QuoteWindowsArg(Utf8ToWide(a));
    }

    STARTUPINFOW startup_info{};
    startup_info.cb = sizeof(startup_info);

    PROCESS_INFORMATION process_info{};
    std::vector<wchar_t> command_buffer(command_line.begin(), command_line.end());
    command_buffer.push_back(L'\0');

    const BOOL created = CreateProcessW(
        exe.c_str(),
        command_buffer.data(),
        nullptr,
        nullptr,
        FALSE,
        0,
        nullptr,
        nullptr,
        &startup_info,
        &process_info
    );
    if (!created) {
        throw std::runtime_error("Failed to spawn child process");
    }

    WaitForSingleObject(process_info.hProcess, INFINITE);

    DWORD exit_code = 0;
    if (!GetExitCodeProcess(process_info.hProcess, &exit_code)) {
        CloseHandle(process_info.hThread);
        CloseHandle(process_info.hProcess);
        throw std::runtime_error("Failed to read child process exit code");
    }

    CloseHandle(process_info.hThread);
    CloseHandle(process_info.hProcess);

    if (exit_code > static_cast<DWORD>(std::numeric_limits<int>::max())) {
        return EXIT_FAILURE;
    }
    return static_cast<int>(exit_code);
}
#else
int SpawnChildAndWait(const std::filesystem::path& exe_path, const std::vector<std::string>& args) {
    if (exe_path.empty()) {
        throw std::runtime_error("Failed to spawn child process: executable path is empty");
    }

    std::vector<std::string> argv_storage;
    argv_storage.reserve(args.size() + 1);
    argv_storage.push_back(exe_path.string());
    argv_storage.insert(argv_storage.end(), args.begin(), args.end());

    std::vector<char*> argvp;
    argvp.reserve(argv_storage.size() + 1);
    for (auto& arg : argv_storage) {
        argvp.push_back(arg.data());
    }
    argvp.push_back(nullptr);

    pid_t child_pid = 0;
    const int spawn_rc = posix_spawnp(
        &child_pid,
        argv_storage[0].c_str(),
        nullptr,
        nullptr,
        argvp.data(),
        environ
    );
    if (spawn_rc != 0) {
        throw std::runtime_error("Failed to spawn child process: " + std::string(std::strerror(spawn_rc)));
    }

    int status = 0;
    while (waitpid(child_pid, &status, 0) == -1) {
        if (errno == EINTR) {
            continue;
        }
        throw std::runtime_error("Failed to wait for child process: " + std::string(std::strerror(errno)));
    }

    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    if (WIFSIGNALED(status)) {
        return 128 + WTERMSIG(status);
    }

    throw std::runtime_error("Child process terminated unexpectedly");
}
#endif

} // namespace cli_process
