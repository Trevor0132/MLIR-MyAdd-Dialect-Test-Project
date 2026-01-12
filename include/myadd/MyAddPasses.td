include "mlir/Pass/PassBase.td"

// 定义MyAddOptPass（对应C++中的MyAddOptPass）
def MyAddOptPass : Pass<"myadd-opt", "::mlir::func::FuncOp"> {
  let summary = "MyAdd dialect optimizations (zero add elimination)";
  let description = "Eliminate myadd.add ops with zero operand (0 + x → x)";
  let constructor = "mlir::myadd::createMyAddOptPass()";
}