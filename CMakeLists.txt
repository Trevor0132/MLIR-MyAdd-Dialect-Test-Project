cmake_minimum_required(VERSION 3.20)
project(MyAddDialect LANGUAGES CXX)

# ===================== 核心配置：指定MLIR/LLVM路径 =====================
# 替换为你本地的LLVM编译目录（步骤1中llvm-project-16.0.0/build）
set(LLVM_BUILD_DIR "D:/llvm-project/build")
# 替换为你本地的LLVM源码目录
set(LLVM_SRC_DIR "D:/llvm-project")

# 查找MLIR和LLVM的CMake配置
find_package(MLIR REQUIRED CONFIG PATHS ${LLVM_BUILD_DIR}/lib/cmake/mlir)
find_package(LLVM REQUIRED CONFIG PATHS ${LLVM_BUILD_DIR}/lib/cmake/llvm)

# ===================== 编译配置 =====================
# 添加头文件路径
include_directories(
  ${PROJECT_SOURCE_DIR}/include          # 本地myadd头文件
  ${MLIR_INCLUDE_DIRS}                   # MLIR头文件
  ${LLVM_INCLUDE_DIRS}                   # LLVM头文件
  ${LLVM_SRC_DIR}/mlir/include           # MLIR源码头文件（TableGen需要）
  ${LLVM_BUILD_DIR}/mlir/include         # MLIR编译生成的头文件
)

# 设置C++标准和编译选项
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -Wno-unused-parameter)

# ===================== TableGen生成辅助代码 =====================
# TableGen工具路径（编译后的mlir-tblgen）
set(MLIR_TABLEGEN_EXE ${LLVM_BUILD_DIR}/bin/mlir-tblgen.exe)
set(DIALECT_TABLEGEN_OUTPUT_DIR ${PROJECT_SOURCE_DIR}/include/myadd)

# -------------------------- 修改点1：拆分Dialect生成命令（拆分为.h.inc和.cpp.inc） --------------------------
# 1. 生成MyAddDialect.h.inc（仅声明，--gen-dialect-decls）
add_custom_command(
  OUTPUT ${DIALECT_TABLEGEN_OUTPUT_DIR}/MyAddDialect.h.inc
  COMMAND ${MLIR_TABLEGEN_EXE}
  --gen-dialect-decls                  # MLIR16：生成Dialect声明（.h.inc）
  -I${LLVM_SRC_DIR}/mlir/include
  -I${LLVM_BUILD_DIR}/mlir/include
  -I${PROJECT_SOURCE_DIR}/include
  -dialect myadd                # 指定方言
  -o ${DIALECT_TABLEGEN_OUTPUT_DIR}/MyAddDialect.h.inc  # 仅指定一个输出
  ${PROJECT_SOURCE_DIR}/include/myadd/MyAddOps.td
  DEPENDS ${PROJECT_SOURCE_DIR}/include/myadd/MyAddOps.td
  COMMENT "Generating MyAddDialect.h.inc"
)

# 2. 生成MyAddDialect.cpp.inc（仅定义，--gen-dialect-defs）
add_custom_command(
  OUTPUT ${DIALECT_TABLEGEN_OUTPUT_DIR}/MyAddDialect.cpp.inc
  COMMAND ${MLIR_TABLEGEN_EXE}
  --gen-dialect-defs                  # MLIR16：生成Dialect定义（.cpp.inc）
  -I${LLVM_SRC_DIR}/mlir/include
  -I${LLVM_BUILD_DIR}/mlir/include
  -I${PROJECT_SOURCE_DIR}/include
  -dialect myadd                # 指定方言
  -o ${DIALECT_TABLEGEN_OUTPUT_DIR}/MyAddDialect.cpp.inc  # 仅指定一个输出
  ${PROJECT_SOURCE_DIR}/include/myadd/MyAddOps.td
  DEPENDS ${PROJECT_SOURCE_DIR}/include/myadd/MyAddOps.td
  COMMENT "Generating MyAddDialect.cpp.inc"
)

# -------------------------- 修改点2：拆分Op生成命令（拆分为.h.inc和.cpp.inc） --------------------------
# 3. 生成MyAddOps.h.inc（仅声明，--gen-op-decls）
add_custom_command(
  OUTPUT ${DIALECT_TABLEGEN_OUTPUT_DIR}/MyAddOps.h.inc
  COMMAND ${MLIR_TABLEGEN_EXE}
  --gen-op-decls                      # MLIR16：生成Op声明（.h.inc）
  -I${LLVM_SRC_DIR}/mlir/include
  -I${LLVM_BUILD_DIR}/mlir/include
  -I${PROJECT_SOURCE_DIR}/include
  -dialect myadd                # 指定方言
  -o ${DIALECT_TABLEGEN_OUTPUT_DIR}/MyAddOps.h.inc       # 仅指定一个输出
  ${PROJECT_SOURCE_DIR}/include/myadd/MyAddOps.td
  DEPENDS ${PROJECT_SOURCE_DIR}/include/myadd/MyAddOps.td
  COMMENT "Generating MyAddOps.h.inc"
)

# 4. 生成MyAddOps.cpp.inc（仅定义，--gen-op-defs）
add_custom_command(
  OUTPUT ${DIALECT_TABLEGEN_OUTPUT_DIR}/MyAddOps.cpp.inc
  COMMAND ${MLIR_TABLEGEN_EXE}
  --gen-op-defs                      # MLIR16：生成Op定义（.cpp.inc）
  -I${LLVM_SRC_DIR}/mlir/include
  -I${LLVM_BUILD_DIR}/mlir/include
  -I${PROJECT_SOURCE_DIR}/include
  -dialect myadd                # 指定方言
  -o ${DIALECT_TABLEGEN_OUTPUT_DIR}/MyAddOps.cpp.inc     # 仅指定一个输出
  ${PROJECT_SOURCE_DIR}/include/myadd/MyAddOps.td
  DEPENDS ${PROJECT_SOURCE_DIR}/include/myadd/MyAddOps.td
  COMMENT "Generating MyAddOps.cpp.inc"
)

# 自定义目标：确保TableGen先执行（依赖所有生成的.inc文件）
add_custom_target(MyAddOpsIncGen
  DEPENDS
  ${DIALECT_TABLEGEN_OUTPUT_DIR}/MyAddDialect.h.inc
  ${DIALECT_TABLEGEN_OUTPUT_DIR}/MyAddDialect.cpp.inc
  ${DIALECT_TABLEGEN_OUTPUT_DIR}/MyAddOps.h.inc
  ${DIALECT_TABLEGEN_OUTPUT_DIR}/MyAddOps.cpp.inc
)

# ===================== 编译自定义Dialect库 =====================
# 移除：原有的静态库定义，现在使用下面的SHARED库定义


# ===================== 编译验证程序 =====================
add_executable(test_load test/test_load.cpp)
# 链接自定义Dialect库
target_link_libraries(test_load PRIVATE MyAddDialect)
# 链接MLIR库
target_link_libraries(test_load
  PRIVATE
  MyAddDialect
  MLIRIR
  MLIRParser
  MLIRAsmParser
  MLIRFuncDialect
  MLIRArithDialect
  MLIRSupport
  LLVMSupport
)

# ===================== 编译方言插件 =====================
add_library(MyAddDialectPlugin SHARED
  lib/MyAddDialectPlugin.cpp
)
# 依赖MyAddDialect库
target_link_libraries(MyAddDialectPlugin PRIVATE MyAddDialect)
# 设置插件输出名称（可选）
set_target_properties(MyAddDialectPlugin PROPERTIES
  OUTPUT_NAME "MyAddDialectPlugin"
)

# 新增：编译build_add_ir程序（编程式构建IR）
add_executable(build_add_ir test/build_add_ir.cpp)
# 链接依赖库
target_link_libraries(build_add_ir
  PRIVATE
  MyAddDialect          # 自定义Dialect库
  MLIRIR                # MLIR核心IR库
  MLIRArithDialect      # arith.constant依赖
  MLIRFuncDialect       # func.func/return依赖
  MLIRSupport           # MLIR支持库
  LLVMSupport           # LLVM输出库（llvm::outs）
)
# 头文件路径（确保能找到myadd的头文件）
target_include_directories(build_add_ir
  PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
  ${LLVM_SRC_DIR}/mlir/include
  ${LLVM_BUILD_DIR}/mlir/include
)

# ===================== 补充TableGen生成Pass的辅助代码 =====================
# Pass的TableGen生成
set(PASS_TABLEGEN_OUTPUT_DIR ${PROJECT_SOURCE_DIR}/include/myadd)
add_custom_command(
  OUTPUT ${PASS_TABLEGEN_OUTPUT_DIR}/MyAddPasses.h.inc
  COMMAND ${MLIR_TABLEGEN_EXE}
  --gen-pass-decls
  -I${LLVM_SRC_DIR}/mlir/include
  -I${LLVM_BUILD_DIR}/mlir/include
  -I${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/include/myadd/MyAddPasses.td
  -o ${PASS_TABLEGEN_OUTPUT_DIR}/MyAddPasses.h.inc
  DEPENDS ${PROJECT_SOURCE_DIR}/include/myadd/MyAddPasses.td
)

# 新增Pass的TableGen目标
add_custom_target(MyAddPassesIncGen
  DEPENDS ${PASS_TABLEGEN_OUTPUT_DIR}/MyAddPasses.h.inc
)

# ===================== 修改MyAddDialect库，添加Pattern/Pass源文件 =====================
add_library(MyAddDialect STATIC
  lib/MyAddOps.cpp
  lib/MyAddPatterns.cpp  # 新增Pattern文件
  lib/MyAddPasses.cpp    # 新增Pass文件
  lib/MyAddToLLVM.cpp    # 新增转换文件
  ${DIALECT_TABLEGEN_OUTPUT_DIR}/MyAddDialect.cpp.inc
  ${DIALECT_TABLEGEN_OUTPUT_DIR}/MyAddOps.cpp.inc
  ${PASS_TABLEGEN_OUTPUT_DIR}/MyAddPasses.h.inc # Pass的辅助代码
)

# 依赖Pass的TableGen目标
add_dependencies(MyAddDialect MyAddOpsIncGen MyAddPassesIncGen)

# 补充Pattern/Pass的依赖库
target_link_libraries(MyAddDialect
  PRIVATE
  MLIRIR
  MLIRParser
  MLIRAsmParser
  MLIRSupport
  LLVMSupport
  MLIRArithDialect      # arith.constant依赖
  MLIRFuncDialect       # func.func遍历依赖
  MLIRPass              # Pass核心库
  MLIRRewrite           # Pattern重写库
)

# ===================== 可选：编译测试优化的程序（test_rewrite.cpp） =====================
add_executable(test_rewrite test/test_rewrite.cpp)
target_link_libraries(test_rewrite
  PRIVATE
  MyAddDialect
  MLIRIR
  MLIRArithDialect
  MLIRFuncDialect
  MLIRPass
  MLIRRewrite
  MLIRSupport
  LLVMSupport
)
target_include_directories(test_rewrite
  PRIVATE
  ${PROJECT_SOURCE_DIR}/include
)

# ===================== 补充转换相关的依赖库 =====================
target_link_libraries(MyAddDialect
  PRIVATE
  # 原有依赖...
  MLIRLLVMDialect       # LLVM Dialect核心库
  MLIRTransforms        # 转换Pass依赖
)

# ===================== 新增：编译转换测试程序（test_llvm_lowering.cpp） =====================
add_executable(test_llvm_lowering test/test_llvm_lowering.cpp)
target_link_libraries(test_llvm_lowering
  PRIVATE
  MyAddDialect
  MLIRIR
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRPass
  MLIRRewrite
  MLIRSupport
  LLVMSupport
  # 新增：MLIR翻译为LLVM IR的库
  MLIRTranslateLib
  MLIRTargetLLVMIRExport
  MLIRLLVMToLLVMIRTranslation  # LLVM方言翻译接口
  # 新增：转换库
  MLIRArithToLLVM
  MLIRFuncToLLVM
)
target_include_directories(test_llvm_lowering
  PRIVATE
  ${PROJECT_SOURCE_DIR}/include
)