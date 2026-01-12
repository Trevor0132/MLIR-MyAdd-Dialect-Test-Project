#include "myadd/MyAddToLLVM.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h" // 导出LLVM IR
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h" // LLVM翻译接口
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
// 修正头文件
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include <iostream>

int main() {
  std::cout << "开始运行test_llvm_lowering程序" << std::endl;
  
  mlir::MLIRContext context;
  mlir::DialectRegistry registry;
  registry.insert<mlir::myadd::MyAddDialect, mlir::arith::ArithDialect, mlir::func::FuncDialect, mlir::LLVM::LLVMDialect>();
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();

  std::cout << "上下文和方言注册完成" << std::endl;

  // 解析测试IR
  const char *irText = R"(
    module {
      func.func @add_test() -> i32 {
        %c1_i32 = arith.constant 1 : i32
        %c2_i32 = arith.constant 2 : i32
        %0 = myadd.add %c1_i32, %c2_i32 : i32
        return %0 : i32
      }
    }
  )";
  auto module = mlir::parseSourceString(irText, &context);
  if (!module) {
    llvm::errs() << "❌ 解析IR失败！\n";
    return 1;
  }

  std::cout << "✅ IR解析成功！" << std::endl;
  std::cout << "原始IR：" << std::endl;
  (*module)->print(llvm::outs());
  std::cout << std::endl;

  // 运行转换Pass（将myadd、arith和func全部转换为LLVM）
  mlir::PassManager pm(&context);
  pm.addPass(mlir::myadd::createConvertMyAddToLLVMPass());
  bool passFailed = mlir::failed(pm.run(*module));
  
  std::cout << "转换Pass运行" << (passFailed ? "失败" : "成功") << "！" << std::endl;
  std::cout << "转换后的IR：" << std::endl;
  (*module)->print(llvm::outs());
  std::cout << std::endl;
  
  if (passFailed) {
    llvm::errs() << "❌ 转换Pass运行失败！\n";
    return 1;
  }

  // 注册LLVM方言的翻译接口（必须在导出LLVM IR之前调用）
  mlir::registerLLVMDialectTranslation(context);
  
  // 导出为标准LLVM IR
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(*module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "❌ 导出LLVM IR失败！\n";
    return 1;
  }

  // 打印LLVM IR
  llvm::outs() << "✅ 标准LLVM IR：\n";
  llvmModule->print(llvm::outs(), nullptr);
  return 0;
}

/*
D:/llvm-project/build/bin/llc.exe ^
  -mtriple=x86_64-w64-mingw32 ^
  -filetype=asm ^
  test_add.ll ^
  -o test_add.s
*/