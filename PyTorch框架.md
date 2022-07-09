# PyTorch框架

 [ *基础入门教程编译制作*](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

## PyTorch 基础语法

使用GPU计算。

### 张量

类似于多维数组。

```python
# 我们创建一个  5×3  矩阵。使用 torch.empty 可以返回填充了未初始化数据的张量。张量的形状由可变参数大小定义。
import torch

torch.empty(5, 3)
```

### 操作

```
任何以下划线结尾的操作都会用结果替换原变量。例如：x.copy_(y), x.t_(), 都会改变 x。
```



- 直接在2个张量之间使用‘+’
- 调用方法

torch.add(x, y[, out=result])

- 替换

```python
y.add_(x)  # 将 x 加到 y-----  y += x
y
```



- 索引方式操作类似于Numpy
- torch.view改变张量的维度和大小

```python
x = to
z = x.view(-1, 8)  # size -1 从其他维度推断rch.randn(4, 4)
y = x.view(16)

x.size(), y.size(), z.size()
```



## Autograd 自动求导



## 神经网络分类器

