[TOC]

# 介绍

> NLP最为常用的**深度神经网络类型**：循环神经网络。

## 关系

![](D:\事务\我的事务\拓展学习\笔记\pictures\楼+深度学习\NLP\NLP-ML-DL-AI关系.jpg)

## NLP组成

### 自然语言理解NLU

- 3个方面

1. 词义分析。
2. 句法分析。
3. 语义分析。

### 自然语言生成NLG

> 从结构化数据（NLU分析后的数据）以读取方式自动生成文本。

- 3个阶段

1. 文本规划：完成结构化数据中的基础内容规划。
2. 语句规划：从结构化数据中组合语句来表达信息流。
3. 实现：产生语法通顺的语句来表达文本。

## NLP应用

- 信息检索：对大规模文档进行索引。
- 语音识别：识别包含口语在内的自然语言的声学信号转换成符合预期的信号。
- 机器翻译：将一种语言翻译成另外一种语言。
- 智能问答：自动回答问题。
- 对话系统：通过多回合对话，跟用户进行聊天、回答、完成某项任务。
- 文本分类：将文本自动归类。
- 情感分析：判断某段文本的情感倾向
- 文本生成：根据需求自动生成文本
- 自动文摘：归纳，总结文本的摘要。

# 序列模型

- 含义

序列指这个模型中的**元素不再独立**，而是具有**一定的相关性**。

## 模型分类

![](D:\事务\我的事务\拓展学习\笔记\pictures\楼+深度学习\NLP\序列模型分类-输入输出.jpg)

- 一对一模型：我们之前的模型，比如全连接神经网络，就是一对一的结构。我们给它一个输入，它能得到一个输出，而且不同输入间被视作了独立的关系，分别进行学习或者识别等任务。而现在我们要关注的，是后四幅图，也就是**时序的模型**。
- 一对多模型：根据模型的一个输出，可以用来预测一个序列。比如对于一个图像，输出一串字幕来描述它。
- 多对一模型：根据一个序列输入，从而预测一个值。比如根据用户在一家饭店的留言，判断用户对这家饭店的评价。那么，输入是一段话是一个序列输入，输出是 0 至 5 之间的一个数作为打分。
- **多对多有延迟**：根据模型的序列输入，我们根据已有的输入，**有延迟地输出一个序列**。常见的任务比如翻译的学习任务，英文翻译成中文。模型需要等英文输入到一定程度后，才能翻译成中文，这样输入输出都为一个序列。
- **多对多无延迟**：根据模型的序列输入，根据输入**同步输出**一个序列。常见的比如做天气预报，需要实时根据测得的温度、湿度等做出下雨概率的预测。但是，每一次预测其实也要考虑之前的一些序列输入，而不仅由这一时刻的输入所决定。

序列模型的典型特点是**输入输出不独立**，往往输出跟前一步、甚至前几步的输入相关。

# 简单RNN

循环神经网络的解决方法，是通过让隐藏层不仅只考虑上一层的输出，还包括了**上一时刻该隐藏层的输出**。理论上，循环神经网络能够包含**前面任意多个时刻的状态**。实践中为了降低模型的复杂性，我们一般只处理前面几个状态的输出。

相比于传统的前馈式神经网络，循环神经网络**将不同时序的同一层前后连了起来**，权值上就会**通过引入另一个权值**，来**决定上一时刻的输出如何作为输入**，并**影响到下一时刻的输出**。

## 基本结构

![](D:\事务\我的事务\拓展学习\笔记\pictures\楼+深度学习\NLP\RNN基本结构.jpg)

前馈式神经网络的输入 𝑥0⋯𝑥𝑡 是一次性传入的，而循环神经网络是 𝑥0 传入后得到 ℎ0 之后，再进行 𝑥1 的输入。

- 详细展开

![](D:\事务\我的事务\拓展学习\笔记\pictures\楼+深度学习\NLP\RNN详细展开.jpg)

上面的公式中添加了激活函数，例如我们想求得 ℎ2，实际上简单来写：
$$
h_2 = l_1(x_1) + r_1(h_1) \tag{1}
$$

# LSTM长短期记忆模型

> 简单RNN无法解决长依赖问题。
>
> 为此，RNN可以通过改变模块，来增强隐藏层的功能，**`LSTM`** 和 **`GRU`** 最为经典。

## 含义

LSTM 全称是 [ *长短期记忆模型*](https://zh.wikipedia.org/zh-hans/長短期記憶)，是目前非常流行的循环神经网络元结构。它与一般的 RNN 结构本质上并没有什么不同，只是使用了**不同的函数去计算隐藏层的状态**。

- 门结构

1. 遗忘门。
2. 输入门。
3. 输出门。

这里，我们再举一个更形象的例子重新梳理一下 LSTM 不同门结构的作用。例如需要训练一个语言模型，LSTM 单元状态中都应该包含当前主语的性别信息，这样才能正确预测并使用人称代词。但是，当语句开始使用新的主语时，遗忘门的作用就是把上文中的主语性别给忘掉避免对下文造成影响。接下来，输入门就需要把新的主语性别信息传入到单元中。

最后，输出门的作用是对最终输出进行过滤。假设模型刚刚接触了一个代词，接下来可能要输出一个动词，而这就与代词的信息相关。比如这个动词应该采用单数还是复数形式，这就需要把刚学到的和代词相关的信息都加入到元状态中来，才能得到正确的输出。

# 双向RNN

> 在经典的循环神经网络中，状态的传输是**从前往后单向进行**的。然而在有些问题中，当前时刻的输出不仅和之前的状态有关系，也**和之后的状态相关**。就比如我们平常的**中英文翻译**，我们根据前后文才能让我们的翻译更加准确。这时就需要双向 RNN（Bidirectional Recurrent Neural Networks）来解决这类问题。

原始的双向 RNN 是由两个相对简单的 RNN 上下叠加在一起组成的。输出由这两个 RNN 的状态共同决定。

# 深度RNN

深度 RNN 则是通过叠加多个 RNN 网络构建更复杂的模型

# RNN搭建

- 词索引

数据集中的每一条评论都经过预处理，并编码为词索引（整数）的序列表示。词索引的意思是，将词按数据集中出现的频率进行索引，例如整数 3 编码了数据中第三个最频繁的词。

- 预处理

如果你输出多条评论后，你会发现每条评论的长度大小不一。但是神经网络输入时，我们必须保证每一条数据的形状是一致的，所以这里需要对数据进行预处理。

这里使用 `tf.keras.preprocessing.sequence.pad_sequences()` 进行处理，通过指定最大长度 `maxlen` 达到裁切向量的目的。同时，如果原句子长度不足，将会在头部通过 0 填充。

- 词嵌入

词嵌入，英文叫 Word Embedding，这是一种十分常用的**词索引特征化手段**。

关于 Embedding，这里举一个简单的例子用来理解。例如单词 apple 对应的词索引为 100。通过 Embedding 转化后，100 就可以变成一个指定大小的向量，比如转化为 [1,2,1][1,2,1]。其中，1 表示 apple 是个很讨人喜欢的东西，2 表示 apple 是个水果，最后的 1 表示 apple 是有益身体健康的，这就是一个特征化的过程。

```python
# IMDB数据集
import numpy as np
import tensorflow as tf

# 加载数据, num_words 表示只考虑最常用的 n 个词语，代表本次所用词汇表大小
MAX_DICT = 1000
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(
    num_words=MAX_DICT)

# 看到原评论内容就需要通过索引从字典中找到原单词
index = tf.keras.datasets.imdb.get_word_index()  # 获取词索引表
reverse_index = dict([(value, key) for (key, value) in index.items()])
comment = " ".join([reverse_index.get(i - 3, "#")
                    for i in X_train[0]])  # 还原第 1 条评论
'''
上面的输出语句中，部分用 # 代替的词即不包含在这 1000 常用词之中。
'''

MAX_LEN = 200  # 设定句子最大长度
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, MAX_LEN)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, MAX_LEN)

# 有了 Embedding 结构，我们就可以搭建一个简单的全连接网络来完成评论情绪分类了。
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(MAX_DICT, 16, input_length=MAX_LEN))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.summary()
```



```python
'''
开始使用 Keras 构建顺序模型结构，该循环神经网络分为 3 层，分别是 Embedding，SimpleRNN，以及用于输出的 Dense 全连接层
'''
model_RNN = tf.keras.Sequential()
model_RNN.add(tf.keras.layers.Embedding(MAX_DICT, 32))
# dropout 是层与层之前的 dropout 数值，recurrent_dropout 是上个时序与这个时序的 dropout 值
model_RNN.add(tf.keras.layers.SimpleRNN(units=32,
                                        dropout=0.2,
                                        recurrent_dropout=0.2))
model_RNN.add(tf.keras.layers.Dense(1, activation='sigmoid'))
'''
model_RNN.suammry() 可以帮我们清晰地看出模型结构，模型总共要学的参数数量较大。接下来，我们对模型进行编译和训练，并最终输出模型在测试集上的评估情况。
'''
model_RNN.compile(optimizer='Adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
model_RNN.fit(X_train, y_train, BATCH_SIZE, EPOCHS,
              validation_data=(X_test, y_test))
```

```python
'''
下面，我们把上面的 SimpleRNN 结构更换为 LSTM 结构，TensorFlow 中可以直接调用 tf.keras.layers.LSTM  来实现。API 参数上与 SimpleRNN 近乎相同，这里就不再赘述了。
'''
model_LSTM = tf.keras.Sequential()
model_LSTM.add(tf.keras.layers.Embedding(MAX_DICT, 32))
model_LSTM.add(tf.keras.layers.LSTM(units=32,
                                    dropout=0.2,
                                    recurrent_dropout=0.2))
model_LSTM.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model_LSTM.summary()

model_LSTM.compile(optimizer='Adam', loss='binary_crossentropy',
                   metrics=['accuracy'])
model_LSTM.fit(X_train, y_train, BATCH_SIZE, EPOCHS,
               validation_data=(X_test, y_test))
```



```python
model_GRU = tf.keras.Sequential()
model_GRU.add(tf.keras.layers.Embedding(MAX_DICT, 32))
model_GRU.add(tf.keras.layers.GRU(units=32,
                                  dropout=0.2,
                                  recurrent_dropout=0.2))
model_GRU.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model_GRU.summary()

model_GRU.compile(optimizer='Adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
model_GRU.fit(X_train, y_train, BATCH_SIZE, EPOCHS,
              validation_data=(X_test, y_test))
```

