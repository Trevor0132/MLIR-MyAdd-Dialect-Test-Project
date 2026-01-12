#ifndef MYADD_MYADDPASSES_H
#define MYADD_MYADDPASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace myadd {

// 声明 Pass 创建函数
std::unique_ptr<mlir::Pass> createMyAddOptPass();

} // namespace myadd
} // namespace mlir

#endif // MYADD_MYADDPASSES_H