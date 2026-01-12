#include "myadd/MyAddDialect.h"
#include "myadd/MyAddOps.h"
// MLIR核心头文件
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Builders.h"  // 修正：OpBuilder 在 Builders.h 中
#include "mlir/IR/BuiltinOps.h"  // 修正：ModuleOp 在 BuiltinOps.h 中
#include "mlir/IR/Location.h"
// 依赖的Dialect头文件（arith/func）
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
// LLVM输出头文件
#include "llvm/Support/raw_ostream.h"

int main() {
  // ===================== 步骤1：初始化MLIR上下文和注册器 =====================
  mlir::MLIRContext context;
  mlir::DialectRegistry registry;
  
  // 注册所有需要的Dialect（必须！否则构建Op时会报错）
  registry.insert<mlir::myadd::MyAddDialect>();  // 自定义AddDialect
  registry.insert<mlir::arith::ArithDialect>();   // 常量Dialect（arith.constant）
  registry.insert<mlir::func::FuncDialect>();     // 函数Dialect（func.func/return）
  
  context.appendDialectRegistry(registry);
  // 加载Dialect（确保上下文能识别这些Dialect）
  context.loadDialect<mlir::myadd::MyAddDialect, mlir::arith::ArithDialect, mlir::func::FuncDialect>();

  // ===================== 步骤2：创建OpBuilder（IR构建器） =====================
  // UnknownLoc：临时位置（调试用，生产环境可用FileLineColLoc指定文件/行号）
  mlir::Location loc = mlir::UnknownLoc::get(&context);
  mlir::OpBuilder builder(&context);

  // ===================== 步骤3：创建MLIR模块（顶层容器） =====================
  auto module = builder.create<mlir::ModuleOp>(loc);
  // 设置模块的名称（可选，方便识别）
  module.setName("add_1_plus_2_module");

  // ===================== 步骤4：创建函数（计算入口） =====================
  // 函数类型：无输入，输出i32（对应1+2的结果类型）
  mlir::Type i32Type = builder.getI32Type();
  mlir::FunctionType funcType = builder.getFunctionType({}, {i32Type});
  // 创建func.func Op（函数名：add_1_plus_2，类型：funcType）
  auto func = builder.create<mlir::func::FuncOp>(loc, "add_1_plus_2", funcType);
  // 为函数添加入口基本块（函数必须有至少一个基本块）
  auto entryBlock = func.addEntryBlock();
  // 将builder的插入点设置到入口基本块的起始位置（后续Op会插入到这里）
  builder.setInsertionPointToStart(entryBlock);

  // ===================== 步骤5：创建常量Op（1和2） =====================
  // 常量1：i32类型，值为1
  auto const1 = builder.create<mlir::arith::ConstantOp>(
    loc,
    builder.getI32IntegerAttr(1)  // LLVM 16.0.0：直接传整数属性，自动推导类型
  );
  // 常量2：i32类型，值为2
  auto const2 = builder.create<mlir::arith::ConstantOp>(
    loc,
    builder.getI32IntegerAttr(2)
  );

  // ===================== 步骤6：创建自定义myadd.add Op（1+2） =====================
  // 构建AddOp：输入const1和const2，输出i32类型
  auto addOp = builder.create<mlir::myadd::AddOp>(
    loc,          // 位置信息
    i32Type,      // 输出类型（必须和输入一致，否则verify报错）
    const1.getResult(),  // 左操作数（常量1）
    const2.getResult()   // 右操作数（常量2）
  );

  // ===================== 步骤7：创建返回Op（输出结果） =====================
  builder.create<mlir::func::ReturnOp>(loc, addOp.getResult());

  // ===================== 步骤8：将函数添加到模块，验证并打印IR =====================
  // 将函数添加到模块（模块是函数的容器）
  module.push_back(func);
  
  // 验证整个模块的合法性（触发所有Op的verify方法）
  if (failed(module.verify())) {
    llvm::errs() << "❌ 模块验证失败！\n";
    return 1;
  }

  // 打印完整的MLIR IR文本（可读格式）
  llvm::outs() << "✅ 成功构建MLIR模块，完整IR如下：\n";
  module->print(llvm::outs());

  return 0;
}