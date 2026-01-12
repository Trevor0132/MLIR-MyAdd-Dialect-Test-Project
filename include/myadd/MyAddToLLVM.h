#ifndef MYADD_MYADDTOLLVM_H
#define MYADD_MYADDTOLLVM_H

#include "myadd/MyAddOps.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h" // 转换Pattern核心头文件
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h" // Pass基类

namespace mlir {
namespace myadd {

// 注册myadd Dialect到LLVM Dialect的转换Pattern
void populateMyAddToLLVMConversionPatterns(
    LLVMTypeConverter &converter,
    RewritePatternSet &patterns);

// 创建myadd到LLVM的转换Pass
std::unique_ptr<Pass> createConvertMyAddToLLVMPass();

} // namespace myadd
} // namespace mlir

#endif // MYADD_MYADDTOLLVM_H