#include "myadd/MyAddPatterns.h"
#include "myadd/MyAddPasses.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/raw_ostream.h"
// 修正头文件
#include "mlir/IR/Builders.h"

int main() {
  mlir::MLIRContext context;
  mlir::DialectRegistry registry;
  registry.insert<mlir::myadd::MyAddDialect, mlir::arith::ArithDialect, mlir::func::FuncDialect>();
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();

  // 解析测试IR
  const char *irText = R"(
    module {
      func.func @add_zero_test() -> i32 {
        %c0_i32 = arith.constant 1 : i32
        %c2_i32 = arith.constant 0 : i32
        %0 = myadd.add %c0_i32, %c2_i32 : i32
        return %0 : i32
      }
    }
  )";
  auto module = mlir::parseSourceString(irText, &context);
  if (!module) {
    llvm::errs() << "❌ 解析IR失败！\n";
    return 1;
  }

  // 创建PassManager并添加MyAddOptPass
  mlir::PassManager pm(&context);
  // 添加函数级别的Pass
  pm.addNestedPass<mlir::func::FuncOp>(mlir::myadd::createMyAddOptPass());

  // 运行Pass（执行优化）
  if (failed(pm.run(*module))) {
    llvm::errs() << "❌ 优化Pass运行失败！\n";
    return 1;
  }

  // 打印优化后的IR
  llvm::outs() << "✅ 优化后的IR：\n";
  mlir::OpPrintingFlags flags;
  (*module)->print(llvm::outs(), flags);
  return 0;
}