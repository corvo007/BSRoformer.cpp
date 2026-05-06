#pragma once

// Install crash handlers for fatal signals (SIGSEGV, SIGABRT, etc.) and
// Windows SEH exceptions. On crash, prints a stack trace to stderr before
// terminating. Call once at the top of main().
namespace crash_handler {
void install();
}  // namespace crash_handler
