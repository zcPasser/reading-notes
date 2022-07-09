[TOC]

# TensorFlow基础概念语法

> 从0开始  **自主编程** 构建 **深度神经网络**，过程十分复杂；
>
> 目前在整个深度学习社区中，比较流行的框架有 **TensorFlow** 和 **PyTorch** ;
>
> 其中，**TensorFlow** 背靠 **Google**，且开发者群体庞大，更新和发版速度非常快。

---

## TensorFlow介绍

-  高度灵活性。
- 可移植性。
- 自动求微分。
- 多语言支持。
- 优化计算资源

## 张量

> **tensor** 即为张量。

- 定义1

物理学或传统数学方法，将张量看成1个 **多维数组**，当**变换坐标** 或 **变换基底**时，其分量会按照 一定规则进行变换，规则有2种：**协变** 或 **逆变转换**。

- 定义2

现代数学，将张量定义成某个 **矢量空间** 或其**对偶空间上的多重线性映射**，这矢量空间在需要引入基底之前不固定任何坐标系统。例如**协变矢量**，可以描述为 **1-形式**，或者作为**逆变矢量**的**对偶空间的元素**。

- 通俗定义

0阶张量 - 标量（只有大小）；

1维数组 - 向量 - 一阶张量；

2维数组 - 矩阵 - 二阶张量；

3维数组 - 数据立体 - 三阶张量；

...

N维数组- 张量 - N阶张量；

### 张量类型

- TensorFlow中，每个张量 具备 3个 基础属性：

1. 数据。
2. 数据类型。
3. 形状

- 张量本身分 2 种类型：

1. **`tf.Variable`** ：变量 Tensor，需要**指定初始值**，常用于**定义可变参数**，例如**神经网络的权重**。
2. **`tf.constant`** ：常量 Tensor，需要**指定初始值**，**定义不变化的张量**。

- python实现

```python
# 可以通过传入数组来新建变量和常量类型的张量：
import tensorflow as tf

v = tf.Variable([[1, 2], [3, 4]])  # 形状为 (2, 2) 的二维变量

c = tf.constant([[1, 2], [3, 4]])  # 形状为 (2, 2) 的二维常量

# 张量 3 部分 属性
<tf.Tensor: shape=(2, 2), dtype=int32, numpy=array([[1, 2],[3, 4]], dtype=int32)>
    
# 直接通过 .numpy() 输出张量的 NumPy 数组
c.numpy()
```

### 常用张量

#### 类型

- [ *`tf.zeros`*](https://tensorflow.google.cn/api_docs/python/tf/zeros)：新建指定形状且全为 0 的常量 Tensor
- [ *`tf.zeros_like`*](https://tensorflow.google.cn/api_docs/python/tf/zeros_like)：参考某种形状，新建全为 0 的常量 Tensor
- [ *`tf.ones`*](https://tensorflow.google.cn/api_docs/python/tf/ones)：新建指定形状且全为 1 的常量 Tensor
- [ *`tf.ones_like`*](https://tensorflow.google.cn/api_docs/python/tf/ones_like)：参考某种形状，新建全为 1 的常量 Tensor
- [ *`tf.fill`*](https://tensorflow.google.cn/api_docs/python/tf/fill)：新建一个指定形状且全为某个标量值的常量 Tensor

#### python实现

```python
tf.zeros([3, 3])  # 3x3 全为 0 的常量 Tensor

tf.ones_like(c)  # 与 c 形状一致全为 1 的常量 Tensor

tf.fill([2, 3], 6)  # 2x3 全为 6 的常量 Tensor

tf.linspace(1.0, 10.0, 5)  # 从 1 到 10，共 5 个等间隔数

tf.range(start=1, limit=10, delta=2)  # 从 1 到 10 间隔为 2
```

## Eager Execution

> 1.x中的 **Graph Execution（图与会话机制）** 更改为 **Eager Execution（动态图机制）** 。

- python代码

```python
a = tf.constant([1., 2., 3., 4., 5., 6.], shape=[2, 3])
b = tf.constant([7., 8., 9., 10., 11., 12.], shape=[3, 2])

c = tf.linalg.matmul(a, b)  # 矩阵乘法

tf.linalg.matrix_transpose(c)  # 转置矩阵
```

## 自动微分



```python
# 使用  tf.GradientTape 跟踪全部运算过程，以便在必要的时候计算梯度。
w = tf.Variable([1.0])  # 新建张量

with tf.GradientTape() as tape:  # 追踪梯度
    loss = w * w  # 计算过程

tape.gradient(loss, w)  # 计算梯度

'''
 tf.GradientTape 会像磁带一样记录下计算图中的梯度信息，然后使用 .gradient 即可回溯计算出任意梯度，这对于使用 TensorFlow 低阶 API 构建神经网络时更新参数非常重要。
'''
```

## 常用模块

### 类别

- [ *`tf.`*](https://tensorflow.google.cn/versions/r2.0/api_docs/python/tf)：包含了张量定义，变换等常用函数和类。
- [ *`tf.data`*](https://tensorflow.google.cn/versions/r2.0/api_docs/python/tf/data)：输入数据处理模块，提供了像 `tf.data.Dataset` 等类用于封装输入数据，指定批量大小等。
- [ *`tf.image`*](https://tensorflow.google.cn/versions/r2.0/api_docs/python/tf/image)：图像处理模块，提供了像图像裁剪，变换，编码，解码等类。
- [ *`tf.keras`*](https://tensorflow.google.cn/versions/r2.0/api_docs/python/tf/keras)：原 Keras 框架高阶 API。包含原 `tf.layers` 中高阶神经网络层。
- [ *`tf.linalg`*](https://tensorflow.google.cn/versions/r2.0/api_docs/python/tf/linalg)：线性代数模块，提供了大量线性代数计算方法和类。
- [ *`tf.losses`*](https://tensorflow.google.cn/versions/r2.0/api_docs/python/tf/losses)：损失函数模块，用于方便神经网络定义损失函数。
- [ *`tf.math`*](https://tensorflow.google.cn/versions/r2.0/api_docs/python/tf/math)：数学计算模块，提供了大量数学计算函数。
- [ *`tf.saved_model`*](https://tensorflow.google.cn/versions/r2.0/api_docs/python/tf/saved_model)：模型保存模块，可用于模型的保存和恢复。
- [ *`tf.train`*](https://tensorflow.google.cn/versions/r2.0/api_docs/python/tf/train)：提供用于训练的组件，例如优化器，学习率衰减策略等。
- [ *`tf.nn`*](https://tensorflow.google.cn/versions/r2.0/api_docs/python/tf/nn)：提供用于构建神经网络的底层函数，以帮助实现深度神经网络各类功能层。
- [ *`tf.estimator`*](https://tensorflow.google.cn/versions/r2.0/api_docs/python/tf/estimator)：高阶 API，提供了预创建的 Estimator 或自定义组件。