# pandas

## 官方API

[ *官方文档相应章节*](https://pandas.pydata.org/pandas-docs/stable/reference/io.html) 

## 数据读取

- 读取CSV文件

CSV文件存储时是一个二维表格，可以用来读取CSV文件。

pandas会自动读取为DataFrame类型。

```python
# 我们想要使用 Pandas 来分析数据，那么首先需要读取数据。大多数情况下，数据都来源于外部的数据文件或者数据库。Pandas 提供了一系列的方法来读取外部数据，非常全面。下面，我们以最常用的 CSV 数据文件为例进行介绍。

# 读取数据 CSV 文件的方法是 pandas.read_csv()，你可以直接传入一个相对路径，或者是网络 URL。

https://labfile.oss.aliyuncs.com/courses/906/los_census.csv
df = pd.read_csv("https://labfile.oss.aliyuncs.com/courses/906/los_census.csv")
df

# DataFrame 是 Pandas 构成的核心。一切的数据，无论是外部读取还是自行生成，我们都需要先将其转换为 Pandas 的 DataFrame 或者 Series 数据类型。实际上，大多数情况下，这一切都是设计好的，无需执行额外的转换工作。
```

- pd.read_前缀开始的方法

该类方法可以读取各式各样的数据文件，且支持连接数据库。

## 基本操作

- DataFrame

<!--在DataFrame上的大多方法和技巧都适用于Series-->

1、3部分 = 列名称 + 索引 + 数据。

![image-20220312110612714](C:\Users\zhangcai\AppData\Roaming\Typora\typora-user-images\image-20220312110612714.png)

2、常用方法

可以**同时使用 Pandas 和 NumPy 提供的 API 对同一数据进行操作**，并在二者之间进行随意转换。这就是一个非常灵活的工具生态圈。

```python
# Pandas 提供了 head() 和 tail() 方法，它可以帮助我们只预览一小块数据。
df.head()  # 默认显示前 5 条

df.tail(7)  # 指定显示后 7 条

# Pandas 还提供了统计和描述性方法，方便你从宏观的角度去了解数据集。
# describe() 相当于对数据集进行概览，会输出该数据集每一列数据的计数、最大值、最小值等。
df.describe()

# Pandas 基于 NumPy 开发，所以任何时候你都可以通过 .values 将 DataFrame 转换为 NumPy 数组。
df.values
df.index  # 查看索引
df.columns  # 查看列名
df.shape  # 查看形状
```

## 数据选择

```python
# 数据预处理过程中，我们往往会对数据集进行切分，只将需要的某些行、列，或者数据块保留下来，输出到下一个流程中去。这也就是所谓的数据选择，或者数据索引。

# 由于 Pandas 的数据结构中存在索引、标签，所以我们可以通过多轴索引完成对数据的选择。
```

### 基于索引数字选择

```python
# 新建一个 DataFrame 之后，如果未自己指定行索引或者列对应的标签，那么 Pandas 会默认从 0 开始以数字的形式作为行索引，并以数据集的第一行作为列对应的标签。其实，这里的「列」也有数字索引，默认也是从 0 开始，只是未显示出来。

# 首先可以基于数字索引对数据集进行选择。这里用到的 Pandas 中的 .iloc 方法。该方法可以接受的类型有：
整数。例如：5
整数构成的列表或数组。例如：[1, 2, 3]
布尔数组。
可返回索引值的函数或参数。

# 首先，我们可以选择前 3 行数据。这和 Python 或者 NumPy 里面的切片很相似。
df.iloc[:3]
# 选择特定的一行。
df.iloc[5]
# df.iloc[] 的 [[行]，[列]] 里面可以同时接受行和列的位置
# 选择 2，4，6 行，可以这样做。
df.iloc[[1, 3, 5]]
# 选择第 2-4 列。
df.iloc[:, 1:4]
```

### 基于标签名称选择

```python
# 除了根据数字索引选择，还可以直接根据标签对应的名称选择。这里用到的方法和上面的 iloc 很相似，少了个 i 为 df.loc[]。

df.loc[] 可以接受的类型有：
单个标签。例如：2 或 'a'，这里的 2 指的是标签而不是索引位置。
列表或数组包含的标签。例如：['A', 'B', 'C']。
切片对象。例如：'A':'E'，注意这里和上面切片的不同之处，首尾都包含在内。
布尔数组。
可返回标签的函数或参数。

# 选择前 3 行：
df.loc[0:2]
# 再选择 1，3，5 行：
df.loc[[0, 2, 4]]
# 选择 2-4 列：
df.loc[:, 'Total Population':'Total Males']

# 选择 1，3 行和 Median Age 后面的列：
df.loc[[0, 2], 'Median Age':]
```

PS:切片是首尾，**左右皆闭区间**。

## 数据删减

```python
# Pandas 中，以 .drop 开头的方法都与数据删减有关。

# DataFrame.drop 可以直接去掉数据集中指定的列和行。
# 一般在使用时，我们指定 labels 标签参数，然后再通过 axis 指定按列或按行删除即可。
# 当然，你也可以通过索引参数删除数据，具体查看官方文档。
df.drop(labels=['Median Age', 'Total Males'], axis=1)

# 除此之外，另一个用于数据删减的方法 DataFrame.dropna 也十分常用，其主要的用途是删除缺少值，即数据集中空缺的数据列或行。
```

```python
# DataFrame.drop_duplicates 则通常用于数据去重，即剔除数据集中的重复值。
# 使用方法非常简单，默认情况下，它会根据所有列删除重复的行。也可以使用 subset 指定要删除的特定列上的重复项，要删除重复项并保留最后一次出现，请使用 keep='last'。
```

## 数据填充

### 检测缺失值

```python
# Pandas 为了更方便地检测缺失值，将不同类型数据的缺失均采用 NaN 标记。这里的 NaN 代表 Not a Number，它仅仅是作为一个标记。例外是，在时间序列里，时间戳的丢失采用 NaT 标记。
    
# Pandas 中用于检测缺失值主要用到两个方法，分别是：isna() 和 notna()，故名思意就是「是缺失值」和「不是缺失值」。默认会返回布尔值用于判断。

df = pd.DataFrame(np.random.rand(9, 5), columns=list('ABCDE'))
# 插入 T 列，并打上时间戳
df.insert(value=pd.Timestamp('2017-10-1'), loc=0, column='Time')
# 将 1, 3, 5 列的 2，4，6，8 行置为缺失值
df.iloc[[1, 3, 5, 7], [0, 2, 4]] = np.nan
# 将 2, 4, 6 列的 3，5，7，9 行置为缺失值
df.iloc[[2, 4, 6, 8], [1, 3, 5]] = np.nan
df
# 然后，通过 isna() 或 notna() 中的一个即可确定数据集中的缺失值。
df.isna()

# 实际上，面对缺失值一般就是填充和剔除两项操作。填充和清除都是两个极端。如果你感觉有必要保留缺失值所在的列或行，那么就需要对缺失值进行填充。如果没有必要保留，就可以选择清除缺失值。

# 其中，缺失值剔除的方法 dropna() 已经在上面介绍过了。下面来看一看填充缺失值 fillna() 方法。

# 首先，我们可以用相同的标量值替换 NaN，比如用 0。
df.fillna(0)
# 除了直接填充值，我们还可以通过参数，将缺失值前面或者后面的值填充给相应的缺失值。例如使用缺失值前面的值进行填充：
df.fillna(method='pad')
# 或者是后面的值：
df.fillna(method='bfill')
```

