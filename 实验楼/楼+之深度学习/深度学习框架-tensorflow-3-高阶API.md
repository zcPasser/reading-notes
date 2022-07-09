[TOC]

# tf.keras模块

## 介绍

- 核心数据结构 **`Model`**
- 2种 `模式`

最简单且最常用：**`Sequential顺序模型`**，多个网络层线性堆叠的栈。

复杂结构：**`函数模型`**

## keras顺序模型

- 创建模型

```python

# 创建 `顺序模型`
import tensorflow as tf

model = tf.keras.models.Sequential()  # 定义顺序模型
```

- 添加网络层

keras中神经网络层 全部位于  `tf.keras.layers` 下方。包含各自经典神经网络 需要的 功能层，如 全连接层 `tf.keras.layers.Dense`。

```python
Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

'''
units: 正整数，输出空间维度。
activation: 激活函数。若不指定，则不使用激活函数(即， 线性激活: a(x) = x)。
use_bias: 布尔值，该层是否使用偏置项量。
kernel_initializer: kernel 权值矩阵的初始化器。
bias_initializer: 偏置项量的初始化器.
kernel_regularizer: 运用到 kernel 权值矩阵的正则化函数。
bias_regularizer: 运用到偏置项的正则化函数。
activity_regularizer: 运用到层的输出的正则化函数。
kernel_constraint: 运用到 kernel 权值矩阵的约束函数。
bias_constraint: 运用到偏置项量的约束函数。
'''
```

Dense 实现操作： `output = activation(dot(input, kernel) + bias)` 其中 `activation` 是按逐个元素计算的激活函数，`kernel` 是由网络层创建的权值矩阵，以及 `bias` 是其创建的偏置项量 (只在 `use_bias` 为 `True` 时才有用)。

- 实例

![](D:\事务\我的事务\拓展学习\笔记\pictures\楼+深度学习\tensorflow构建神经网络\拟构建3层神经网络-2层隐含层.png)

```python
# 添加全连接层
model.add(tf.keras.layers.Dense(units=30, activation=tf.nn.relu))  # 输出 30，relu 激活
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))  # 输出 10，softmax 激活
```

一开始我们定义顺序模型就像是「打地基」，而只需要使用 `add` 就可以「盖楼房」了。上面的代码中，激活函数 `tf.nn.relu` 也可以用名称 `'relu'` 代替，即可写作 `tf.keras.layers.Dense(units=30, activation='relu')`。

---

添加完神经网络层之后，就可以使用 `model.compile` 来**编译顺序模型**。这时，需要通过参数指定优化器，损失函数，以及评估方法。

```python
# adam 优化器 + 交叉熵损失 + 准确度评估
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

其中，参数可以使用如上所示的名称或者 TensorFlow 实例。如果使用实例的优化器和损失函数，需要从 TensorFlow 支持的 [ *全部优化器列表*](https://tensorflow.google.cn/api_docs/python/tf/keras/optimizers) 和 [ *全部损失函数列表*](https://tensorflow.google.cn/api_docs/python/tf/losses) 中选择。如果使用名称，需要从 Keras 支持的 [ *名称列表*](https://keras.io/zh/losses/) 中选择。

特别注意的是，这里的损失函数是不能胡乱选的。你需要根**据网络的输出形状**和**真实值的形状**来决定。如果不匹配就会报错。由于实验示例网络最后通过了 `tf.nn.softmax` 激活，那么对于单个样本输入最终就会得到一个概率最大的整数类型的输出。此时就选择了 `sparse_categorical_crossentropy` 具有整数输出类型的交叉熵损失函数。随着实验后续深入，见的越多就会有使用经验了。

---

定义数据，选择 **`DIGITS`** 数据集。

```python
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import numpy as np

# 准备 DIGITS 数据
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=1)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
```

---

定义数据后，使用 `model.fit` 就可以开始模型训练了。

需要指定小批量 `batch_size` 大小，和全数据迭代次数 `epochs` 参数。

```python
# 模型训练
model.fit(X_train, y_train, batch_size=64, epochs=5)
```

```
Train on 1437 samples
Epoch 1/5
1437/1437 [==============================] - 0s 319us/sample - loss: 1.7744 - accuracy: 0.5108
Epoch 2/5
1437/1437 [==============================] - 1s 542us/sample - loss: 1.6410 - accuracy: 0.5428
Epoch 3/5
1437/1437 [==============================] - 0s 223us/sample - loss: 1.5077 - accuracy: 0.5706
Epoch 4/5
1437/1437 [==============================] - 0s 331us/sample - loss: 1.3863 - accuracy: 0.5915
Epoch 5/5
1437/1437 [==============================] - 1s 557us/sample - loss: 1.2613 - accuracy: 0.6235
```

Out[10]:

```
<tensorflow.python.keras.callbacks.History at 0x7fe3ea7ee7d0>
```

----

可以看到每一个 Epoch 之后模型在训练数据上的损失和分类准确度。使用 `model.evaluate` 即可评估训练后模型在测试集上的损失和分类准确度。

```python
# 模型评估
model.evaluate(X_test, y_test)
```

```
360/360 [==============================] - 0s 375us/sample - loss: 1.2152 - accuracy: 0.6028
```

Out[12]:

```
[1.215155045191447, 0.6027778]
```

实际使用过程中，我们一般会直接将测试数据通过 `validation_data` 参数传入训练过程。那么，每一个 Epoch 之后都会同时输出在训练集和测试集上的分类评估结果。

```python
# 使用参数传入测试数据
model.fit(X_train, y_train, batch_size=64, epochs=5,
          validation_data=(X_test, y_test))
```

## keras函数模型

> 除了顺序模型，Keras 也提供函数式 API。和顺序模型最大的不同在于，函数模型可以通过**多输入多输出的方式**。并且**所有的模型都是可调用**的，就像层一样利用函数式模型的接口，我们可以很**容易的重用已经训练好的模型**。

- 函数式API 重写 模型结构

```python
inputs = tf.keras.Input(shape=(64,))  # 输入层
x = tf.keras.layers.Dense(units=30, activation='relu')(inputs)  # 中间层
outputs = tf.keras.layers.Dense(units=10, activation='softmax')(x)  # 输出层

# 函数式 API 需要指定输入和输出
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model
```

- 模型编译 和 训练 和 顺序模型 基本一致

```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=64, epochs=20,
          validation_data=(X_test, y_test))
```

## keras模型存储及推理

- 保存模型目的

1. 对于较大的训练任务，保存模型 可以方便 **后续恢复 重新使用**。
2. 保存后的 模型 可以方便 **模型部署**。

- TensorFlow模型3类要素

1. 模型权重值。
2. 模型配置。
3. 优化器配置。

- 保存实例

只需要保存模型权重值，可以使用 `tf.keras.Model.save_weights` ，并指定存放路径。

```python
model.save_weights('./weights/model')  # 保存检查点名称为 model，路径为 ./weights

model.load_weights('./weights/model')  # 恢复检查点
```

---

**`检查点`**

默认情况下，该方法会以 TensorFlow **检查点文件格式** **保存模型的权重**。检查点文件是 TensorFlow 特有的模型权重保存方法，**其默认**会以**每 10 分钟（600 秒）写入一个检查点**，训练时间较短则**只保存一个检查点**。检查点默认情况下**只保存 5 个**，即模型**训练过程中不同时间点的版本状态**。

我们一般会在大型任务训练时设置检查点保存。这样做的好处在于一旦因为意外情况导致训练终止，TensorFlow 可以**加载检查点状态**，**避免**又需要**从头开始训练**。

---

如果我们需要模型推理，一般情况会使用 `model.save` 保存完整的模型，即包含模型权重值、模型配置乃至优化器配置等。例如，下面将模型存为 Keras HDF5 格式，其为 Keras 多后端实现的默认格式。

```python
model.save('model.h5')  # 保存完整模型
```

接下来，可以使用 `tf.keras.models.load_model` 重载模型。

```python
model_ = tf.keras.models.load_model('model.h5')  # 调用模型
```

---

`model.summary()` 可以用来查看 Keras 模型结构，包含神经网络层和参数等详细数据。

```python
model_.summary()
```

```
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         [(None, 64)]              0         
_________________________________________________________________
dense_6 (Dense)              (None, 30)                1950      
_________________________________________________________________
dense_7 (Dense)              (None, 10)                310       
=================================================================
Total params: 2,260
Trainable params: 2,260
Non-trainable params: 0
___________________________________
```

然后，使用 `predict` 方法就可以完成模型推理了。

```python
preds = model_.predict(X_test[:3])  # 预测前 3 个测试样本
preds
```

```
array([[8.37582537e-09, 9.88358557e-01, 3.29617498e-04, 2.22985864e-06,
        4.02094302e-05, 5.17627647e-08, 9.52130267e-06, 7.86766975e-07,
        1.12549197e-02, 4.15329396e-06],
       [1.28534128e-10, 4.45253105e-11, 5.08233506e-06, 9.54681454e-08,
        4.22154553e-13, 9.99774396e-01, 2.87068806e-11, 1.19944470e-05,
        2.08547150e-04, 4.16083417e-08],
       [9.99985933e-01, 1.07765760e-12, 4.51835054e-12, 8.68503013e-12,
        3.65387304e-10, 6.63992807e-07, 1.29746295e-05, 1.16919641e-10,
        2.10557932e-10, 3.83739859e-07]], dtype=float32)
```

预测结果为神经网络通过 Softmax 激活后的输出值。所以我们通过 NumPy 找出每个样本输出最大概率及其对应的索引，其索引也就是最终的预测目标了。同样，可以输出测试数据真实标签进行对比。

```python
np.argmax(preds, axis=1), np.max(preds, axis=1)  # 找出每个样本预测概率最大值索引及其概率
```

# Estimator高阶API

> Estimator 是 TensorFlow 中的高阶 API，它可以将模型的训练、预测、评估、导出等操作封装在一起，构成一个 Estimator。TensorFlow 也提供了大量的预创建 Estimator ，例如线性回归，提升树分类器，深度神经网络分类器等。

- 使用预创建的Estimator编写TensorFlow程序步骤

1. 创建一个或多个输入函数。
2. 定义模型的特征列。
3. 实例化 Estimator，指定特征列和各种超参数。
4. 在 Estimator 对象上调用一个或多个方法，传递适当的输入函数作为数据的来源。

## 步骤1-数据输入

首先，输入到 Estimator 的训练、评估和预测的数据都必须要通过创建输入函数来完成。

输入函数是返回 `tf.data.Dataset` 对象的函数，此对象会输出下列含有两个元素的元组：

- `features`\- Python 字典，其中：
  - 每个键都是特征的名称。
  - 每个值都是包含此特征所有值的数组。
- `label` - 包含每个样本的标签值的数组。

```python
# 将原来的 NumPy 数组转换为 Pandas 提供的 DataFrame，这样就可以将方便将数据转换输入函数要求的 Python 字典类型。
import pandas as pd

# NumPy 数组转换为 DataFrame，并将特征列名处理成字符串类型
X_train_ = pd.DataFrame(X_train, columns=[str(i) for i in range(64)])
y_train_ = pd.DataFrame(y_train, columns=['class'])  # 标签列名
X_test_ = pd.DataFrame(X_test, columns=[str(i) for i in range(64)])
y_test_ = pd.DataFrame(y_test, columns=['class'])

dict(X_train_).keys()  # 运行使用 dict 将数据处理成输入函数要求的字典类型

'''
直接开始定义数据输入函数 input_fn。
tf.data.Dataset  对象是 TensorFlow 强烈推荐使用的数据管道。
当数据为 Dataset 时，你就可以使用 TensorFlow 提供的一系列方法对数据进行变换，例如打乱采样，重复扩展，小批量输入等。
'''

def input_fn(features, labels, batch_size):
    """数据输入函数"""
    # 将数据转换为 Dataset 对象
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    # 将数据重复及处理成小批量
    dataset = dataset.repeat().batch(batch_size)
    return dataset
'''
tf.data.Dataset.from_tensor_slices 函数创建一个代表数组切片的 tf.data.Dataset。系统会在第一个维度内对该数组进行切片。然后 dataset 执行了 repeat 重复序列操作，这样做得目的是保证后续能迭代更多次，否则当数据一遍轮完之后训练就终止了。repeat() 代表无限扩展，即到我们设定的迭代次数。repeat(5) 则表示重复序列 5 次，样本数据变为原来的 5 倍。接着，我们使用了 batch 每次从数据中取出 batch_size 的小批量进行迭代。
'''
```

## 步骤2-定义模型的特征列

特征列是原始数据和 Estimator 之间的媒介，定义特征列就是告诉 Estimator 哪些是特征，每个特征的数据有什么特点。定义特征列并不是说指定几个字符串那样简单，我们需要利用 TensorFlow 提供的方法创建 Estimator 能够识别的特征列。

下面，我们将特征 DataFrame 的列名取出来，并使用 `tf.feature_column.numeric_column` 将其转换为特征列。该方法即告诉 Estimator 特征是 Numeric 数值类型。更多类型的特征列可以参考官方文档 。

```python
feature_columns = []
for key in X_train_.keys():  # 取出 DataFrame 列名
    feature_columns.append(tf.feature_column.numeric_column(key=key))  # 创建数值特征列

feature_columns[:3]  # 查看前 3 个特征列
```

```
[NumericColumn(key='0', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None),
 NumericColumn(key='1', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None),
 NumericColumn(key='2', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None)
```

## 步骤3-实例化

使用 `tf.estimator.DNNClassifier`，需要传入特征列并定义隐含层神经元数量即目标值标签数量。

```python
classifier = tf.estimator.DNNClassifier(
    # 特征列
    feature_columns=feature_columns,
    # 两个隐含层分别为 30 和 10 个神经元
    hidden_units=[30, 10],
    # 模型最终标签类别为 10
    n_classes=10)
```

## 步骤4-在Estimator对象上调用方法

传递适当的输入函数作为数据的来源。值得注意的是，这里将 `input_fn` 调用封装在 `lambda` 中以获取参数。

`steps` 参数告知方法在训练多步后停止训练。`steps` 和先前的 Epoch 不一样，此时相当于取出 `steps` 个 `batch_size` 的数据用于训练。而整个训练过程等价于 `steps * batch_size / 数据总量` 个 Epoch。 所以，通过 `steps` 换算的 Epoch 可能不是整数，但这并不会影响到训练过程。

```python
classifier.train(
    input_fn=lambda: input_fn(X_train_, y_train_, batch_size=64),
    steps=2000)
```



上方训练执行的过程，权重会被自动存为检查点 `.ckpt` 文件。同时，后续的训练过程只有在 `loss` 更优时，检查点才会被覆盖。这样做的原因在于，后续的模型推理需要重载检查点权重，这样能保证存放的检查点性能状态最优。

使用测试数据对模型进行推理评估。此时，我们需要重新定义数据输入函数 `evaluate_input_fn`，原因在于之前定义的输入函数 `input_fn` 执行了 `repeat()` 操作，如果沿用就会导致推理无限持续下去。

```python
def evaluate_input_fn(features, labels, batch_size):
    """评估数据输入函数"""
    # 将数据转换为 Dataset 对象
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    # 将仅处理成小批量
    dataset = dataset.batch(batch_size)
    return dataset
```

最终，使用 `evaluate` 评估模型，传入数据的过程和训练时相似。Estimator 会自动重载训练保存的检查点，并对测试数据进行推理和评估。

```python
# 评估模型
eval_result = classifier.evaluate(
    input_fn=lambda: evaluate_input_fn(X_test_, y_test_, batch_size=64))

print('最终评估准确度：{:.3f}'.format(eval_result['accuracy']))
```

---

- 官方推荐流程

1. 假设存在合适的预创建的 Estimator，使用它构建第一个模型并使用其结果确定基准。
2. 使用此预创建的 Estimator 构建和测试整体管道，包括数据的完整性和可靠性。
3. 如果存在其他合适的预创建的 Estimator，则运行实验来确定哪个预创建的 Estimator 效果最好。
4. 可以通过构建自定义 Estimator 进一步改进模型。

# 神经网络搭建方法小结

## 3种方法

- 利用 `tf.nn` 模块提供的各种神经网络组件和函数。
- 利用 `tf.keras` 模块提供的各种高阶神经网络层。
- 利用 `tf.estimator` 提供的高阶预创建或自定义封装模型。

## 使用场景

- 需要实现的**网络自定义程度**较高，有很多自己的想法且并没有合适的高阶 API 层供调用，那么首选肯定是 `tf.nn`。`tf.nn` 功能强大，但你需要自行定义训练迭代过程，且大多数过程都需要利用 TensorFlow 一系列低阶 API 完成。
- `tf.keras` 模块主要面向于实现包含标准化层的神经网络，例如后面会学习的经典卷积神经网络结构等。API 使用方便，简洁明了。
- `tf.estimator` 本身在 `tf.keras` 之上构建而成，Keras 模型也可以通过创建 Estimator 进行训练 。Estimator 简化了在模型开发者之间共享实现的过程，其可以在本地主机上或分布式多服务器环境中运行基于 Estimator 的模型，而无需更改模型。但 Estimator 的使用可能需要对 TensorFlow 足够熟悉之后才能得心应手。

