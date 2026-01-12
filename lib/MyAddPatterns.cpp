#include "myadd/MyAddPatterns.h"
#include "myadd/MyAddOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h" // arith.constant依赖
#include "mlir/IR/Value.h"
// 修正：用户补充的头文件（确保OpBuilder/BuiltinOps可用）
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;
using namespace mlir::myadd;

// 核心实现：匹配+替换逻辑
LogicalResult AddZeroEliminationPattern::matchAndRewrite(
    Operation *op, PatternRewriter &rewriter) const {
  // 步骤1：将通用Operation转换为具体的AddOp（类型安全检查）
  auto addOp = dyn_cast<AddOp>(op);
  if (!addOp) return failure(); // 不是AddOp，匹配失败

  // 步骤2：获取AddOp的两个操作数
  Value lhs = addOp.getLhs();
  Value rhs = addOp.getRhs();

  // 步骤3：定义辅助函数：判断一个Value是否是常量0
  auto isConstantZero = [&](Value value) -> bool {
    // 检查Value是否是arith.constant Op的结果
    auto constantOp = value.getDefiningOp<arith::ConstantOp>();
    if (!constantOp) return false;
    // 检查常量值是否为0（i32类型）
    auto intAttr = constantOp.getValue().dyn_cast<IntegerAttr>();
    return intAttr && intAttr.getValue() == 0;
  };

  // 步骤4：匹配条件：任意一个操作数是常量0
  Value nonZeroOperand;
  if (isConstantZero(lhs)) {
    nonZeroOperand = rhs; // 左操作数是0，保留右操作数
  } else if (isConstantZero(rhs)) {
    nonZeroOperand = lhs; // 右操作数是0，保留左操作数
  } else {
    return failure(); // 没有0值操作数，匹配失败
  }

  // 步骤5：替换逻辑：删除AddOp，用nonZeroOperand替换其结果
  // rewriter.replaceOp：将addOp的所有使用处替换为nonZeroOperand，并删除addOp
  rewriter.replaceOp(addOp, nonZeroOperand);
  return success(); // 匹配+替换成功
}

// 注册Pattern到PatternSet
void mlir::myadd::populateMyAddRewritePatterns(RewritePatternSet &patterns) {
  // 添加零值消除Pattern到集合中
  patterns.add<AddZeroEliminationPattern>(patterns.getContext());
}