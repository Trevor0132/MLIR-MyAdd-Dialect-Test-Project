# MLIR MyAdd Dialect Test Project

## 项目描述

这是一个 MLIR（Multi-Level Intermediate Representation）测试项目，实现了一个自定义的 "myadd" 方言。该方言包含一个简单的加法操作，并演示了从 MLIR IR 到 LLVM IR 的完整 lowering 流程，最终生成可执行文件。

## 主要功能

- 自定义 MLIR 方言（myadd），实现加法操作
- 从 myadd 方言到 LLVM IR 的转换
- 常量折叠优化
- 从 MLIR 生成可执行文件

## 依赖项

- LLVM/MLIR（需要从源码构建）
- CMake
- GCC（Windows 下使用 MinGW）

## 环境搭建

### 1. 克隆 LLVM 项目

首先，从 GitHub 克隆 LLVM 项目（包含 MLIR）：

```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
```

### 2. 编译 LLVM/MLIR

#### Windows (MinGW)

```bash
# 创建构建目录
mkdir build
cd build

# 配置 CMake（使用 Ninja 或 Make）
cmake -G "MinGW Makefiles" ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_TARGETS_TO_BUILD="X86" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_INSTALL_PREFIX=<install-path>

# 编译（需要较长时间）
mingw32-make -j4
```

#### Linux

```bash
# 创建构建目录
mkdir build
cd build

# 配置 CMake
cmake -G "Unix Makefiles" ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_TARGETS_TO_BUILD="X86" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_INSTALL_PREFIX=<install-path>

# 编译
make -j$(nproc)
```

**注意**：
- `<install-path>` 是安装路径，例如 `D:/llvm-install`
- 编译时间较长，请耐心等待
- 确保 CMake 版本 >= 3.13.4
- Windows 下需要 MinGW-w64 编译器

### 3. 设置环境变量

编译完成后，将 LLVM 安装路径添加到系统 PATH 中，或在构建项目时指定 LLVM_DIR。

## 构建步骤

1. 下载或克隆项目到本地

2. 创建构建目录：
   ```bash
   mkdir build
   cd build
   ```

3. 使用 CMake 配置项目（需要指定 LLVM 路径）：
   ```bash
   cmake .. -DLLVM_DIR=<path-to-llvm-build>/lib/cmake/llvm
   ```

4. 构建项目：
   ```bash
   mingw32-make  # Windows 下使用 MinGW
   # 或在 Linux 下使用 make
   ```

## 运行测试

### 测试 MLIR 到 LLVM IR 转换
```bash
mingw32-make test_llvm_lowering
.\test_llvm_lowering.exe
```

### 生成可执行文件
```bash
# 生成汇编代码
D:/llvm-project/build/bin/llc.exe -mtriple=x86_64-w64-mingw32 -filetype=asm test_add.ll -o test_add.s

# 编译为可执行文件
gcc test_add.s -o test_add.exe

# 运行
.\test_add.exe
```

## MLIR Pass 说明

本项目实现了自定义的 MLIR Pass 来处理 myadd 方言到 LLVM IR 的转换：

### ConvertMyAddToLLVMPass

这是核心的转换 Pass，负责将 myadd 方言的操作转换为 LLVM IR：

- **位置**: `lib/MyAddToLLVM.cpp`
- **功能**:
  - 将 `myadd.add` 操作转换为 `llvm.add`
  - 处理 `func.return` 操作，确保正确的操作数传递
  - 使用 `LLVMTypeConverter` 进行类型转换
  - 设置 `LLVMConversionTarget` 定义合法操作

### 自定义转换模式

项目实现了以下自定义转换模式：

#### ReturnOpToLLVMOpPattern
- **继承**: `ConvertOpToLLVMPattern<func::ReturnOp>`
- **作用**: 正确处理 `func.return` 操作的操作数转换
- **优先级**: 设置为较高优先级（100），确保优先于 MLIR 默认模式
- **解决的问题**: 修复了 `llvm.return` 期望 1 个操作数但实际没有的错误

### Pass 注册和使用

在 `test_llvm_lowering.cpp` 中：
```cpp
PassManager pm(&context);
pm.addPass(createConvertMyAddToLLVMPass());
```

这个 Pass 被添加到 PassManager 中，与其他标准 Pass 一起执行完整的 lowering 流程。

## 项目结构

```
mlir_test/
├── CMakeLists.txt          # 构建配置文件
├── include/
│   └── myadd/             # 方言头文件
│       ├── MyAddDialect.h
│       ├── MyAddOps.h
│       └── MyAddOps.td
├── lib/
│   └── MyAddOps.cpp       # 方言实现
├── test/
│   ├── test_load.cpp      # 基本测试
│   └── test_llvm_lowering.cpp  # LLVM lowering 测试
└── build/                 # 构建目录（自动生成）
    ├── test_add.ll        # 生成的 LLVM IR
    ├── test_add.s         # 生成的汇编代码
    └── test_add.exe       # 生成的可执行文件
```

## 技术细节

- 使用 MLIR 的 Dialect Conversion 框架实现 lowering
- 自定义转换模式处理 func.return 操作
- 支持常量折叠优化
- 生成标准的 LLVM IR 和可执行文件

## 许可证

此项目仅用于学习和测试目的。