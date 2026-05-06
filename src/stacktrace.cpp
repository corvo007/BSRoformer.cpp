#include "stacktrace.h"

#include <sstream>
#include <iomanip>
#include <cstdint>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <dbghelp.h>
#pragma comment(lib, "dbghelp.lib")
#elif defined(__GNUC__) || defined(__clang__)
#include <execinfo.h>
#include <cxxabi.h>
#include <cstdlib>
#endif

namespace stacktrace {

#ifdef _WIN32

static bool g_symbols_initialized = false;

static void init_symbols() {
  if (!g_symbols_initialized) {
    SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
    SymInitialize(GetCurrentProcess(), nullptr, TRUE);
    g_symbols_initialized = true;
  }
}

std::vector<StackFrame> capture(int skip_frames) {
  init_symbols();

  std::vector<StackFrame> result;
  void* stack[64];
  HANDLE process = GetCurrentProcess();

  WORD frames = CaptureStackBackTrace(skip_frames + 1, 64, stack, nullptr);

  char symbol_buffer[sizeof(SYMBOL_INFO) + MAX_SYM_NAME * sizeof(TCHAR)];
  SYMBOL_INFO* symbol = reinterpret_cast<SYMBOL_INFO*>(symbol_buffer);
  symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
  symbol->MaxNameLen = MAX_SYM_NAME;

  IMAGEHLP_LINE64 line;
  line.SizeOfStruct = sizeof(IMAGEHLP_LINE64);

  for (WORD i = 0; i < frames; ++i) {
    StackFrame frame;
    frame.address = stack[i];

    DWORD64 displacement64 = 0;
    if (SymFromAddr(process, reinterpret_cast<DWORD64>(stack[i]), &displacement64, symbol)) {
      frame.function = symbol->Name;
    } else {
      std::ostringstream ss;
      ss << "0x" << std::hex << reinterpret_cast<uintptr_t>(stack[i]);
      frame.function = ss.str();
    }

    DWORD displacement32 = 0;
    if (SymGetLineFromAddr64(process, reinterpret_cast<DWORD64>(stack[i]), &displacement32, &line)) {
      frame.file = line.FileName;
      frame.line = static_cast<int>(line.LineNumber);
    }

    result.push_back(std::move(frame));
  }

  return result;
}

#elif defined(__GNUC__) || defined(__clang__)

static std::string demangle(const char* name) {
  int status = 0;
  char* demangled = abi::__cxa_demangle(name, nullptr, nullptr, &status);
  if (status == 0 && demangled) {
    std::string result(demangled);
    free(demangled);
    return result;
  }
  return name ? name : "";
}

std::vector<StackFrame> capture(int skip_frames) {
  std::vector<StackFrame> result;
  void* stack[64];

  int frames = backtrace(stack, 64);
  char** symbols = backtrace_symbols(stack, frames);

  for (int i = skip_frames + 1; i < frames && symbols; ++i) {
    StackFrame frame;
    frame.address = stack[i];

    std::string sym(symbols[i]);

    size_t start = sym.find('(');
    size_t end = sym.find('+');
    if (start != std::string::npos && end != std::string::npos && end > start) {
      std::string mangled = sym.substr(start + 1, end - start - 1);
      if (!mangled.empty()) {
        frame.function = demangle(mangled.c_str());
      } else {
        frame.function = sym;
      }
    } else {
      frame.function = sym;
    }

    result.push_back(std::move(frame));
  }

  if (symbols) free(symbols);
  return result;
}

#else

std::vector<StackFrame> capture(int /*skip_frames*/) {
  return {};
}

#endif

std::string format(const std::vector<StackFrame>& frames) {
  std::ostringstream ss;
  ss << "Stack trace:\n";

  for (size_t i = 0; i < frames.size(); ++i) {
    const auto& f = frames[i];
    ss << "  #" << std::setw(2) << i << " ";

    if (!f.function.empty()) {
      ss << f.function;
    } else {
      ss << "<unknown>";
    }

    if (!f.file.empty()) {
      ss << "\n       at " << f.file;
      if (f.line > 0) {
        ss << ":" << f.line;
      }
    } else if (f.address) {
      ss << " [0x" << std::hex << reinterpret_cast<uintptr_t>(f.address) << std::dec << "]";
    }

    ss << "\n";
  }

  return ss.str();
}

std::string capture_string(int skip_frames) {
  return format(capture(skip_frames + 1));
}

}  // namespace stacktrace
