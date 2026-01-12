#ifndef MYADD_MYADDOPS_H
#define MYADD_MYADDOPS_H

#include "MyAddDialect.h"
#include "mlir/IR/OpDefinition.h"

// 1. 引入MLIR的Op定义辅助宏
#define GET_OP_CLASSES
#include "MyAddOps.h.inc"

#endif // MYADD_MYADDOPS_H