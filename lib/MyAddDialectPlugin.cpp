#include "mlir/IR/DialectRegistry.h"
#include "myadd/MyAddDialect.h"

// 定义DialectPlugin类（如果头文件不存在，我们手动定义）
namespace mlir {
class DialectPlugin {
public:
  virtual ~DialectPlugin() = default;
  virtual void registerDialects(DialectRegistry &registry) const = 0;
};
} // namespace mlir

// 插件类：继承DialectPlugin
class MyAddDialectPlugin : public mlir::DialectPlugin {
public:
  // 注册方言的方法
  void registerDialects(mlir::DialectRegistry &registry) const override {
    // 注册我们的MyAddDialect
    registry.insert<mlir::myadd::MyAddDialect>();
  }
};

// 导出插件函数：mlir-opt会调用此函数加载插件
extern "C" mlir::DialectPlugin *mlirGetDialectPlugin() {
  return new MyAddDialectPlugin();
}