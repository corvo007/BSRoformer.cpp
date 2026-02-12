# GGML Dependency Management

This project uses a flexible GGML dependency resolution strategy that supports both standalone builds and integration into larger projects.

## Dependency Resolution Strategy

The CMake configuration resolves GGML dependencies in the following priority order:

1. **Existing Target** - If `ggml` target already exists (e.g., from parent project like `whisper.cpp`)
2. **Installed Package** - Search for GGML via `find_package()`
3. **Submodule** - Use `ggml/` subdirectory (if exists)
4. **Sibling Directory** - Use `../ggml` (if exists)
5. **Explicit Path** - Use path specified by `GGML_DIR` variable

## Usage Scenarios

### Scenario 1: Standalone Build

Clone GGML as submodule:
```bash
cd BSRoformer.cpp
git submodule add https://github.com/ggerganov/ggml.git
git submodule update --init --recursive
cmake -B build -DGGML_CUDA=ON
```

Or use sibling directory:
```bash
cd ..
git clone https://github.com/ggerganov/ggml.git
cd BSRoformer.cpp
cmake -B build -DGGML_CUDA=ON
```

### Scenario 2: Shared GGML with whisper.cpp

When both projects need GGML, let whisper.cpp provide it:

```cmake
# Parent Project CMakeLists.txt
cmake_minimum_required(VERSION 3.16)
project(MyParentProject)

# Build whisper.cpp first (includes ggml)
add_subdirectory(whisper.cpp)

# Build BSRoformer - it will reuse whisper.cpp's ggml target
add_subdirectory(BSRoformer.cpp)
```

Or use explicit path:
```bash
cmake -B build -DGGML_DIR=/path/to/shared/ggml
```

### Scenario 3: Custom GGML Location

```bash
cmake -B build -DGGML_DIR=/custom/path/to/ggml
```

## Project Structure Examples

**Option A: Submodule** (Recommended)
```
BSRoformer.cpp/
├── ggml/                    # Git submodule
├── src/
├── tests/
└── CMakeLists.txt
```

**Option B: Sibling Directory**
```
parent/
├── ggml/                    # Shared GGML
├── BSRoformer.cpp/
└── whisper.cpp/             # Also uses ../ggml
```

**Option C: Parent Project**
```
MyProject/
├── external/
│   └── ggml/
├── whisper.cpp/
├── BSRoformer.cpp/
└── CMakeLists.txt          # Defines ggml target
```

## Benefits

- ✅ **Standalone**: Works independently without parent project
- ✅ **Reusable**: Shares GGML across multiple projects
- ✅ **Flexible**: Supports multiple directory layouts
- ✅ **Build Time**: Avoids duplicate GGML compilation
- ✅ **Disk Space**: Single GGML copy for multiple projects
