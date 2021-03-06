# 第6章：面向对象程序设计

## 6.1：类的定义与使用

- 关键字class。
- 派生类。
- 成员。
- 实例化以及调用成员。
- 测试
  - isinstance
  - type
- 占位
  - pass
  - .\_\_doc\__

## 6.2：数据成员与成员方法

- 私有成员
  - 2个或更多下划线开头但不以2个或更多下划线结束的为私有成员，否则不是私有成员。
  - 外部特殊访问：对象名._类名__xxx。
- 公有成员
- 成员访问符
- 下划线
  - _xxx：保护成员，只有类对象、子类对象可访问。
  - \_\_xxx\__：系统定义的特殊成员。
  - __xxx：私有成员。
- 数据成员
  - 对象数据成员
    - self
  - 类数据成员
- 方法
  - 一般指与特定实例绑定的函数，通过对象调用方法时，对象本身将被作为第一个参数自动传递过去。而函数要传递对象，如内置函数sorted之类。
  - 静态方法、类方法通过对象名、类名调用。
- 类与对象的动态性、混入机制
  - 动态为自定义类、对象增加数据成员和成员方法。

## 6.3：继承、多态

- 继承
  - 子类继承父类公有成员，私有成员不继承。
  - 调用父类方法通过super()或基类名.方法名。
- 多态
  - 基类的同一个方法在不同派生类对象中具有不同的表现和行为。

## 6.4：特殊方法与运算符重载

