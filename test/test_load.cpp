#include "myadd/MyAddDialect.h"
#include "myadd/MyAddPasses.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/DialectRegistry.h"

#include "myadd/MyAddOps.h" 
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Parser/Parser.h"  // 添加解析器头文件
#include "mlir/Pass/PassManager.h"  // 添加PassManager头文件

#include <iostream>

int main(int argc, char** argv) {
  // 1. 创建MLIR上下文
  mlir::MLIRContext context;
  // 2. 注册自定义Dialect
  mlir::DialectRegistry registry;
  registry.insert<mlir::myadd::MyAddDialect>();
  context.appendDialectRegistry(registry);
  
  // 3. 加载myadd Dialect
  context.loadDialect<mlir::myadd::MyAddDialect, mlir::func::FuncDialect>();
  
  // 4. 验证是否加载成功
  auto dialect = context.getLoadedDialect("myadd");
  if (dialect) {
    std::cout << "✅ MyAddDialect加载成功！\n" << std::endl;
  } else {
    std::cout << "❌ MyAddDialect加载失败！\n" << std::endl;
    return 1;
  }

  // 检查命令行参数
  if (argc > 1) {
    // 从文件读取和解析IR
    std::string filename = argv[1];
    std::cout << "📖 读取文件: " << filename << std::endl;
    
    // 解析MLIR文件
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceFile<mlir::ModuleOp>(filename, &context);
    if (!module) {
      std::cerr << "❌ 解析文件失败: " << filename << std::endl;
      return 1;
    }
    
    std::cout << "✅ 文件解析成功！" << std::endl;
    
    // 打印原始IR
    std::cout << "📄 原始IR：" << std::endl;
    module->print(llvm::outs());
    std::cout << std::endl;
    
    // 创建PassManager并运行优化
    mlir::PassManager pm(&context);
    pm.addNestedPass<mlir::func::FuncOp>(mlir::myadd::createMyAddOptPass());
    
    std::cout << "🔧 运行MyAdd优化Pass..." << std::endl;
    if (mlir::failed(pm.run(*module))) {
      std::cerr << "❌ 优化Pass运行失败！" << std::endl;
      return 1;
    }
    std::cout << "✅ 优化Pass运行成功！" << std::endl;
    
    // 打印优化后的IR
    std::cout << "✨ 优化后的IR：" << std::endl;
    module->print(llvm::outs());
    std::cout << "\n✅ IR打印成功！" << std::endl;
  } else {
    // 如果没有提供文件参数，则编程式构建IR
    std::cout << "🏗️ 没有提供文件参数，编程式构建IR..." << std::endl;
    
    mlir::OpBuilder builder(&context);
    auto module = builder.create<mlir::ModuleOp>(builder.getUnknownLoc()); // 模块
    builder.setInsertionPointToStart(module.getBody()); // 设置插入点到模块体
    
    auto funcType = builder.getFunctionType({}, {builder.getI32Type()});   // 函数类型：无输入，输出i32
    auto func = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), "add_test", funcType);
    auto entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // 构建常量1和2
    auto const1 = builder.create<mlir::arith::ConstantOp>(builder.getUnknownLoc(), builder.getI32Type(), builder.getI32IntegerAttr(1));
    auto const2 = builder.create<mlir::arith::ConstantOp>(builder.getUnknownLoc(), builder.getI32Type(), builder.getI32IntegerAttr(2));

    // 构建myadd.add Op
    auto addOp = builder.create<mlir::myadd::AddOp>(builder.getUnknownLoc(), builder.getI32Type(), const1, const2);

    // 返回结果
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), addOp.getResult());

    // 打印IR文本（验证Print逻辑）
    module->print(llvm::outs());
    std::cout << "\n✅ IR打印成功！" << std::endl;
  }

  return 0;
}

