#include "myadd/MyAddPatterns.h"
#include "mlir/Pass/Pass.h" // Pass核心头文件
#include "mlir/Dialect/Func/IR/FuncOps.h" // func.func遍历依赖
#include "mlir/Dialect/Arith/IR/Arith.h" // arith.constant 依赖

namespace mlir {
namespace myadd {

// 定义Pass：遍历func.func中的所有Op，应用MyAdd的Pattern
struct MyAddOptPass : public mlir::PassWrapper<MyAddOptPass, mlir::OperationPass<mlir::func::FuncOp>> {
  // Pass的名称（mlir-opt通过--myadd-opt调用）
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MyAddOptPass)
  StringRef getArgument() const override { return "myadd-opt"; }
  StringRef getDescription() const override { return "MyAdd dialect optimizations (zero add elimination)"; }

  // 核心方法：运行Pass
  void runOnOperation() override {
    // 获取当前处理的函数
    mlir::func::FuncOp func = getOperation();

    // 手动遍历函数体中的所有操作并应用零值消除
    func.walk([&](mlir::Operation *op) {
      // 只处理 myadd.add 操作
      if (auto addOp = mlir::dyn_cast<mlir::myadd::AddOp>(op)) {
        // 检查是否是 0 + x 或 x + 0 的形式
        auto lhs = addOp.getLhs();
        auto rhs = addOp.getRhs();

        // 检查左操作数是否是常量 0
        mlir::Value replacement;
        if (isConstantZero(lhs)) {
          replacement = rhs;
        } else if (isConstantZero(rhs)) {
          replacement = lhs;
        } else {
          return; // 不是零值加法，继续
        }

        // 替换操作：将 addOp 的所有使用替换为 nonZeroOperand，然后删除 addOp
        addOp.replaceAllUsesWith(replacement);
        addOp.erase();
      }
    });
  }

private:
  // 辅助函数：检查一个 Value 是否是常量 0
  bool isConstantZero(mlir::Value value) const {
    auto constantOp = value.getDefiningOp<mlir::arith::ConstantOp>();
    if (!constantOp) return false;
    auto intAttr = constantOp.getValue().dyn_cast<mlir::IntegerAttr>();
    return intAttr && intAttr.getValue() == 0;
  }
};

// 注册Pass到MLIR系统
std::unique_ptr<mlir::Pass> createMyAddOptPass() {
  return std::make_unique<MyAddOptPass>();
}

} // namespace myadd
} // namespace mlir

// 注册Pass的宏（MLIR 16.0.0标准用法）
#define GEN_PASS_DEF_MYADDOPTPASS
#include "myadd/MyAddPasses.h.inc"