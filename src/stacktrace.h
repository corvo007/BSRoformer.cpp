#pragma once

#include <string>
#include <vector>

// Cross-platform stack trace utility
namespace stacktrace {

struct StackFrame {
  std::string function;
  std::string file;
  int line = 0;
  void* address = nullptr;
};

// Capture current stack trace (skips the specified number of frames)
std::vector<StackFrame> capture(int skip_frames = 0);

// Format stack trace as a string for display
std::string format(const std::vector<StackFrame>& frames);

// Convenience: capture and format in one call
std::string capture_string(int skip_frames = 1);

}  // namespace stacktrace
