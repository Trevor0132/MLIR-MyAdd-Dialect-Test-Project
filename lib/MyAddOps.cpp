#include "myadd/MyAddOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

// 1. 包含Dialect的实现
#define GET_DIALECT_DEFS
#include "myadd/MyAddDialect.cpp.inc"

// 2. 实现initialize方法：注册操作
void mlir::myadd::MyAddDialect::initialize() {
  // 注册AddOp到这个方言
  addOperations<AddOp>();
}

// 验证 AddOp 的合法性
mlir::LogicalResult mlir::myadd::AddOp::verify() {
  // TableGen 已经确保了操作数和结果的数量，这里主要验证类型一致性

  // 检查所有操作数和结果都是整数类型，并且类型一致
  auto resultType = getResult().getType();
  for (auto operand : getOperands()) {
    if (!operand.getType().isIntOrIndex()) {
      return emitError("myadd.add op operands must be integer types");
    }
    if (operand.getType() != resultType) {
      return emitError("myadd.add op operand type must match result type");
    }
  }

  if (!resultType.isIntOrIndex()) {
    return emitError("myadd.add op result must be integer type");
  }

  return mlir::success();
}

// 3. 生成Op的辅助代码（和TableGen对应）
#define GET_OP_CLASSES
#include "myadd/MyAddOps.cpp.inc"