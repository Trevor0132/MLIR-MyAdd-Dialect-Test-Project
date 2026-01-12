// 1. 引入基础依赖
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"

// 2. 定义MyAddDialect（对应C++的MyAddDialect类）
def MyAddDialect : Dialect {
  let name = "myadd";          // Dialect名字（必须和C++中一致）
  let cppNamespace = "::mlir::myadd"; // C++命名空间
  let useFoldAPI = kEmitFoldAdaptorFolder; // 使用新的折叠适配器
}

// 3. 定义myadd.add Op（整数加法算子）
def AddOp : Op<MyAddDialect, "add", [Commutative]> {
  // 3.1 Op的描述（可选，用于文档）
  let summary = "Integer addition operation";
  let description = "Add two integers and return the result";

  // 3.2 定义输入（两个整数）
  let arguments = (ins I32:$lhs, I32:$rhs);

  // 3.3 定义输出（一个整数，类型和输入一致）
  let results = (outs I32:$result);

  // 定义MLIR汇编格式
  // 格式含义：%左操作数, %右操作数 : 类型
  let assemblyFormat = "$lhs `,` $rhs `:` type($result) attr-dict";

  // 启用自定义验证器
  let hasVerifier = 1;
}