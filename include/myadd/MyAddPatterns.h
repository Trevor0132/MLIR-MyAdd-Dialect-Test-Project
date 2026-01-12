#ifndef MYADD_MYADDPATTERNS_H
#define MYADD_MYADDPATTERNS_H

#include "myadd/MyAddOps.h"
#include "mlir/IR/PatternMatch.h" // Pattern重写核心头文件

namespace mlir {
namespace myadd {

// 定义零值加法消除的Pattern（继承RewritePattern）
struct AddZeroEliminationPattern : public mlir::RewritePattern {
  // 构造函数：指定匹配的Op类型（AddOp）、受益值（越高越优先）、MLIR上下文
  AddZeroEliminationPattern(mlir::MLIRContext *context)
      : RewritePattern(AddOp::getOperationName(), /*benefit=*/1, context) {}

  // 核心方法：匹配+替换逻辑（matchAndRewrite）
  mlir::LogicalResult matchAndRewrite(
      mlir::Operation *op,                // 匹配到的Op（myadd.add）
      mlir::PatternRewriter &rewriter) const override;
};

// 注册Pattern到PatternRewriter的PatternSet中
void populateMyAddRewritePatterns(mlir::RewritePatternSet &patterns);

} // namespace myadd
} // namespace mlir

#endif // MYADD_MYADDPATTERNS_H