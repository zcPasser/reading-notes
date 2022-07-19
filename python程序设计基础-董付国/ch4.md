# 第4章：程序控制结构

## 4.1：条件表达式

- python中除了下列对象外，都可以等价于true
  - false
  - 0、0.0、0j
  - 空值none。
  - 空列表。
  - 空元组。
  - 空集合。
  - 空字典。
  - 空字符串。
  - 空range、空其他迭代对象。

- 不使用 赋值运算符“=”。
- 关系运算符
  - 连续使用。
- 逻辑运算符

## 4.2：选择结构

- 单分支

- 双分支

  - 三元运算符（可嵌套）

  ```python
  value1 if condition else value2
  ```

- 多分支

## 4.3：循环结构

- 含有else子句。

循环自然结束的执行else子句，而因为执行break语句提前结束的不执行子句。

- break
- continue
- 循环代码优化
  - 减少循环内部不必要计算。
  - 循环内部尽量引用局部变量，略快于全局变量。

## 4.3：精彩案例

- 获取当前日期

```python
import time

date = time.localtime()
year, month, day = date[:3]
```

# 习题

- 4.5、

```python
"""
1、20个 随机整数 列表。
2、偶数下标 降序排序，剩余 不变。（切片）

"""

import random

nums = [random.randint(0, 100) for i in range(20)]

nums[::2] = sorted(nums[::2], reverse=False)
```

- 4.6、

```python
"""
因式分解
1、输入 1个整数。
2、因式分解。
"""

num = int(input('输入 1个 整数：'))

res = []
t, i = num, 2

while True:
    if t == 1:
        break
    if t % i == 0:
        res.append(i)
        t //= i
    else:
        i += 1

print(num, '=', '*'.join(map(str, res)))
```

