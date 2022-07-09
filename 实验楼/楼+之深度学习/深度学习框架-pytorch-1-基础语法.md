[TOC]

# PyTorch基础概念语法

## 介绍

|       Package（包）        |                     Description（描述）                      |
| :------------------------: | :----------------------------------------------------------: |
|          `torch`           |   张量计算组件, 兼容 NumPy 数组，且具备强大的 GPU 加速支持   |
|      `torch.autograd`      | 自动微分组件, 是 PyTorch 的核心特点，支持 torch 中所有可微分的张量操作 |
|         `torch.nn`         |     深度神经网络组件, 用于灵活构建不同架构的深度神经网络     |
|       `torch.optim`        | 优化计算组件, 囊括了 SGD, RMSProp, LBFGS, Adam 等常用的参数优化方法 |
|  `torch.multiprocessing`   |     多进程管理组件，方便实现相同数据的不同进程中共享视图     |
|       `torch.utils`        |          工具函数组件，包含数据加载、训练等常用函数          |
| `torch.legacy(.nn/.optim)` |                向后兼容组件, 包含移植的旧代码                |

## 张量类型和定义

|       数据类型 dtype       |       CPU 张量       |         GPU 张量          |
| :------------------------: | :------------------: | :-----------------------: |
|        32-bit 浮点         | `torch.FloatTensor`  | `torch.cuda.FloatTensor`  |
|        64-bit 浮点         | `torch.DoubleTensor` | `torch.cuda.DoubleTensor` |
|     16-bit 半精度浮点      |         N/A          |  `torch.cuda.HalfTensor`  |
|  8-bit 无符号整形(0~255)   |  `torch.ByteTensor`  |  `torch.cuda.ByteTensor`  |
| 8-bit 有符号整形(-128~127) |  `torch.CharTensor`  |  `torch.cuda.CharTensor`  |
|     16-bit 有符号整形      | `torch.ShortTensor`  | `torch.cuda.ShortTensor`  |
|     32-bit 有符号整形      |  `torch.IntTensor`   |  `torch.cuda.IntTensor`   |
|     64-bit 有符号整形      |  `torch.LongTensor`  |  `torch.cuda.LongTensor`  |

其中，默认的 `torch.Tensor` 类型为 `32-bit 浮点`，也就是 `torch.FloatTensor`。

```python
import torch as t

t.Tensor().dtype

# t.set_default_tensor_type('torch.DoubleTensor')
t.Tensor().dtype
t.set_default_tensor_type('torch.DoubleTensor')
t.Tensor().dtype
```

- 创建



```python
# 传入 列表
t.Tensor([1, 2, 3])

# 
```

