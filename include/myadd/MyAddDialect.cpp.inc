/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::myadd::MyAddDialect)
namespace mlir {
namespace myadd {

MyAddDialect::MyAddDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<MyAddDialect>()) {
  
  initialize();
}

MyAddDialect::~MyAddDialect() = default;

} // namespace myadd
} // namespace mlir
