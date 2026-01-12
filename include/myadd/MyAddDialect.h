#ifndef MYADD_MYADDDIALECT_H
#define MYADD_MYADDDIALECT_H

#include "mlir/IR/Dialect.h"

// 1. 定义Dialect的命名空间（和C++命名空间对应）
namespace mlir {
namespace myadd {

// 2. TableGen会生成MyAddDialect类的完整定义，包括initialize方法

} // namespace myadd
} // namespace mlir

// 3. MLIR的注册宏：把Dialect注册到MLIR系统中
#define GET_DIALECT_DECLS
#include "MyAddDialect.h.inc"

#endif // MYADD_MYADDDIALECT_H