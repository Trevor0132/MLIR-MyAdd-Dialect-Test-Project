/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace myadd {

class MyAddDialect : public ::mlir::Dialect {
  explicit MyAddDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~MyAddDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("myadd");
  }
};
} // namespace myadd
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::myadd::MyAddDialect)
