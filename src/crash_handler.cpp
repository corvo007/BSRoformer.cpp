#include "crash_handler.h"
#include "stacktrace.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

static LONG WINAPI unhandled_exception_handler(EXCEPTION_POINTERS* info) {
  const char* desc = "Unknown exception";
  switch (info->ExceptionRecord->ExceptionCode) {
    case EXCEPTION_ACCESS_VIOLATION:       desc = "Access Violation (SIGSEGV)"; break;
    case EXCEPTION_STACK_OVERFLOW:         desc = "Stack Overflow"; break;
    case EXCEPTION_INT_DIVIDE_BY_ZERO:     desc = "Integer Divide by Zero"; break;
    case EXCEPTION_FLT_DIVIDE_BY_ZERO:     desc = "Float Divide by Zero"; break;
    case EXCEPTION_ILLEGAL_INSTRUCTION:    desc = "Illegal Instruction (SIGILL)"; break;
    case EXCEPTION_ARRAY_BOUNDS_EXCEEDED:  desc = "Array Bounds Exceeded"; break;
    case EXCEPTION_IN_PAGE_ERROR:          desc = "In Page Error"; break;
    case EXCEPTION_DATATYPE_MISALIGNMENT:  desc = "Datatype Misalignment"; break;
    case STATUS_HEAP_CORRUPTION:           desc = "Heap Corruption"; break;
  }

  fprintf(stderr, "\n========== FATAL CRASH ==========\n");
  fprintf(stderr, "Exception: %s (0x%08lX)\n", desc,
          static_cast<unsigned long>(info->ExceptionRecord->ExceptionCode));
  fprintf(stderr, "Address:   0x%p\n",
          reinterpret_cast<void*>(info->ExceptionRecord->ExceptionAddress));
  fprintf(stderr, "\n%s", stacktrace::capture_string(0).c_str());
  fprintf(stderr, "=================================\n");
  fflush(stderr);

  return EXCEPTION_CONTINUE_SEARCH;
}

namespace crash_handler {
void install() {
  SetUnhandledExceptionFilter(unhandled_exception_handler);
}
}  // namespace crash_handler

#else  // POSIX

#include <signal.h>
#include <unistd.h>

static void signal_handler(int sig) {
  const char* desc = "Unknown signal";
  switch (sig) {
    case SIGSEGV: desc = "SIGSEGV (Segmentation Fault)"; break;
    case SIGABRT: desc = "SIGABRT (Abort)"; break;
    case SIGFPE:  desc = "SIGFPE (Floating Point Exception)"; break;
    case SIGBUS:  desc = "SIGBUS (Bus Error)"; break;
    case SIGILL:  desc = "SIGILL (Illegal Instruction)"; break;
  }

  // Use write() for async-signal-safety where possible, but we need
  // stacktrace formatting which uses malloc — acceptable since we're dying.
  fprintf(stderr, "\n========== FATAL CRASH ==========\n");
  fprintf(stderr, "Signal: %s (%d)\n", desc, sig);
  fprintf(stderr, "\n%s", stacktrace::capture_string(2).c_str());
  fprintf(stderr, "=================================\n");
  fflush(stderr);

  // Re-raise with default handler to get core dump
  signal(sig, SIG_DFL);
  raise(sig);
}

namespace crash_handler {
void install() {
  struct sigaction sa;
  memset(&sa, 0, sizeof(sa));
  sa.sa_handler = signal_handler;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = SA_RESETHAND;  // One-shot: restore default after firing

  sigaction(SIGSEGV, &sa, nullptr);
  sigaction(SIGABRT, &sa, nullptr);
  sigaction(SIGFPE,  &sa, nullptr);
  sigaction(SIGBUS,  &sa, nullptr);
  sigaction(SIGILL,  &sa, nullptr);
}
}  // namespace crash_handler

#endif
