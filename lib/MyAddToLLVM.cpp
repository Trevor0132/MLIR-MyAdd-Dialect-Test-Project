#include "myadd/MyAddToLLVM.h"
#include "myadd/MyAddOps.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;
using namespace mlir::myadd;

// 自定义func.return到llvm.return的转换Pattern
struct ReturnOpToLLVMOpPattern : public ConvertOpToLLVMPattern<func::ReturnOp> {
  using ConvertOpToLLVMPattern<func::ReturnOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      func::ReturnOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (op.getNumOperands() == 0) {
      rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, ValueRange{});
      return success();
    }
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

// 核心：myadd.add → llvm.add的转换Pattern
struct AddOpToLLVMOpPattern : public ConversionPattern {
  AddOpToLLVMOpPattern(LLVMTypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(AddOp::getOperationName(), 1, context),
        typeConverter(typeConverter) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto addOp = cast<AddOp>(op);
    Value lhs = operands[0];
    Value rhs = operands[1];
    Type resultType = typeConverter.convertType(addOp.getType());
    if (!resultType)
      return failure();
    auto llvmAddOp = rewriter.create<LLVM::AddOp>(
        addOp.getLoc(), resultType, lhs, rhs);
    rewriter.replaceOp(addOp, llvmAddOp.getResult());
    return success();
  }

private:
  LLVMTypeConverter &typeConverter;
};

// 注册转换Pattern到PatternSet
void mlir::myadd::populateMyAddToLLVMConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add(std::make_unique<AddOpToLLVMOpPattern>(converter, patterns.getContext()));
  patterns.add(std::make_unique<ReturnOpToLLVMOpPattern>(converter, 1));
}

// 定义转换Pass
struct ConvertMyAddToLLVMPass
    : public PassWrapper<ConvertMyAddToLLVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertMyAddToLLVMPass)

  StringRef getArgument() const override { return "convert-myadd-to-llvm"; }
  StringRef getDescription() const override { return "Convert MyAdd Dialect to LLVM Dialect"; }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext &context = getContext();

    LLVMTypeConverter typeConverter(&context);
    RewritePatternSet patterns(&context);
    mlir::myadd::populateMyAddToLLVMConversionPatterns(typeConverter, patterns);
    arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);
    // Add our ReturnOpToLLVMOpPattern with higher priority
    patterns.add(std::make_unique<ReturnOpToLLVMOpPattern>(typeConverter, 1));

    LLVMConversionTarget target(context);
    target.addLegalOp<ModuleOp>();

    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

// 创建Pass实例
std::unique_ptr<Pass> mlir::myadd::createConvertMyAddToLLVMPass() {
  return std::make_unique<ConvertMyAddToLLVMPass>();
}