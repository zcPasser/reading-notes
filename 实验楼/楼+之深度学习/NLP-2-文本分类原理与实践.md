[TOC]

# 介绍

- 文本分类流程

文本分词。

特征提取。

构建文本分类器。

# 中文文本分词

- 3个方法

机械分词方法、统计分词方法、二者综合方法。

## 机械分词方法

又称 **`基于规则的分词方法`** 。

这种分词方法按照一定的规则将待处理的字符串与一个词表词典中的词进行逐一匹配，若在词典中找到某个字符串，则切分，否则不切分。

机械分词方法按照匹配规则的方式，又可以分为：正向最大匹配法，逆向最大匹配法和双向匹配法三种。

### 正向最大匹配法

正向最大匹配法（Maximum Match Method，简称：MM）是指从左向右按最大原则与词典里面的词进行匹配。假设词典中最长词是 𝑚m 个字，那么从待切分文本的最左边取 𝑚m 个字符与词典进行匹配，如果匹配成功，则分词。如果匹配不成功，那么取 𝑚−1m−1 个字符与词典匹配，一直取直到成功匹配为止。

- 代码

```python
t = '我们是共产主义的接班人'
d = ('我们', '是', '共产主义', '的', '接班', '人', '你', '我', '社会', '主义')

# 该函数可以获取给定词典长度最大词的长度
def get_max_len(d):
    max_len_word = 0
    for key in d:
        if len(key) > max_len_word:
            max_len_word = len(key)
    return max_len_word

get_max_len(d)

# 按照上方给出的正向最大匹配法伪代码来构建正向最大匹配分词函数。
def mm(t, d):
    words = []  # 用于存放分词结果
    while len(t) > 0:  # 句子长度大于 0，则开始循环
        word_len = get_max_len(d)
        for i in range(0, word_len):
            word = t[0: word_len]  # 取出文本前 word_len 个字符
            if word not in d:  # 判断 word 是否在词典中
                word_len -= 1  # 不在则以 word_len - 1
                word = []  # 清空 word
            else:  # 如果 word 在词典当中
                t = t[word_len:]  # 更新文本起始位置
                words.append(word)
                word = []
    return words
```



### 逆向最大匹配法

逆向最大匹配法（ Reverse Maximum Match Method，简称：RMM）的原理与正向法基本相同，唯一不同的就是切分的方向与正向最大匹配法相反。逆向法从文本末端开始匹配，每次用末端的最长词长度个字符进行匹配。

因为基本原理与正向最大匹配法一样，反向来进行匹配就行。所以这里对算法不再赘述。由于汉语言存在偏正短语，因此逆向匹配法相比与正向匹配的精确度会高一些。

### 双向最大匹配法

双向最大匹配法（Bi-direction Matching Method，简称：BMM）则是将正向匹配法得到的分词结果与逆向匹配法得到的分词结果进行比较，然后按照最大匹配原则，选取次数切分最少的作为结果。

## 基于统计规则的中文分词法

简单来讲，假设我们已经有一个由很多个文本组成的的语料库 D，现在需要对 **我有一个苹果** 进行分词。其中，两个相连的字 **苹** 和 **果** 在不同的文本中连续出现的次数越多，就说明这两个相连字很可能构成 **苹果**。

当字连续组合的概率高过一个临界值时，就认为该组合构成了一个词语。

基于统计的分词，一般情况下首先需要建立统计语言模型。然后再对句子进行单词划分，并对划分结果进行概率计算。最终，获得概率最大的分词方式。这里，我们一般会用到隐马可夫，条件随机场等方法。

## 结巴中文分词

一款基于统计的分词工具，其基于隐马可夫模型设计，并使用了 Viterbi 动态规划算法。

### 三种分词模型

- 精确模式：试图将句子最精确地切开，适合文本分析。（cut默认）
- 全模式：把句子中所有的可以成词的词语都扫描出来，速度非常快，但是不能解决歧义。
- 搜索引擎模式：在精确模式的基础上，对长词再次切分，提高召回率，适合用于搜索引擎分词。

# 英文文本分词

直接通过空格或者标点来将文本进行分开。

对于日常的英文文本分词，我们可以采用下面的代码。例如，`string.punctuation` 提供了常用英文标点符号。

```
'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
```

接下来，删除句子中的标点符号。这里，可以使用 Python 提供了一个名为 `translate()` 的函数，它可以将一组字符映射到另一组字符。我们可以使用函数 `maketrans()` 来创建映射表。我们可以创建一个空的映射表，`maketrans()` 函数的第三个参数允许我们列出在 `translate` 过程中要删除的所有字符。代码如下：

```python
text = """
[English] is a West Germanic language that was first spoken in early 
medieval England and eventually became a global lingua franca.
It is named after the <Angles>, one of the Germanic tribes that 
migrated to the area of Great Britain that later took their name, 
as England.
"""

words = text.split()
print(words)
table = str.maketrans('', '', string.punctuation)
print(table)
stripped = [w.translate(table) for w in words]
print(stripped)
```

# 文本特征提取

分词之后的语料数据是无法直接用于分类的，还需要我们从中**提取特征**，并将这些**文本特征**变换为**数值特征**。只有**向量化后的数值**才能够传入到分类器中训练文本分类模型。

## 词袋模型-特征提取方法

[ *词袋模型*](http://www.di.ens.fr/~josef/publications/sivic09a.pdf)（英语：Bag-of-words model，简称：BoW）是最最基础的一类特征提取方法，其主要思路是忽略掉了文本的语法和语序，用一组无序的单词序列来表达一段文字或者一个文档。可以这样理解，我们把整个**文档集**的所有出现的词都丢进**袋子**里面，**去重后无序排列**。这样，就可以按**照词语出现的次数**来表示文档。

词袋模型的表示方法为，对照词袋，统计原句子中某个单词出现的次数。这样，无论句子的长度如何，均可用等长度的词袋向量进行表示。例如，对于句子 `"Bats can see via echolocation. See the bat sight sneeze!"`，其可以转换为向量 [0,2,1,⋯,0,1,0][0,2,1,⋯,0,1,0]。

可以使用 scikit-learn 提供的 `sklearn.feature_extraction.text.CountVectorizer` 来构建词袋模型。

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "The elephant sneeze at the sight of potato.",
    "Bat can see via echolocation. See the bat sight sneeze!",
    "Wonder, she open the door to the studio.",
]

vectorizer = CountVectorizer()
vectors = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())  # 打印出词袋
vectors.toarray()  # 打印词向量
```

```
['at', 'bat', 'can', 'door', 'echolocation', 'elephant', 'of', 'open', 'potato', 'see', 'she', 'sight', 'sneeze', 'studio', 'the', 'to', 'via', 'wonder']
```

Out[17]:

```
array([[1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 2, 0, 0, 0],
       [0, 2, 1, 0, 1, 0, 0, 0, 0, 2, 0, 1, 1, 0, 1, 0, 1, 0],
       [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 2, 1, 0, 1]])
```

上面的例子中，我们在制作 `corpus` 时已手动对英语单词中单复数和时态词汇做了处理，全部变为原型。这个过程在实际应用时，可以通过预处理代码来完成。

- 小变种

不以单词实际出现的次数表示，而是采取类似独热编码的方式。单词出现即置为 1，未出现即为 0。

这个过程同样可以使用 scikit-learn 来完成。这里用到了 `sklearn.preprocessing.Binarizer` 对上面 `CountVectorizer` 处理结果进行独热编码转换。

```python
from sklearn.preprocessing import Binarizer

freq = CountVectorizer()
corpus_ = freq.fit_transform(corpus)

onehot = Binarizer()
onehot.fit_transform(corpus_.toarray())
```

```
array([[1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0],
       [0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
       [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1]])
```

## TF-IDF模型-特征提取方法

[ *TF-IDF 模型*](https://doi.org/10.1017%2FCBO9781139058452.002)（英语：Term frequency–inverse document frequency）是一种用于信息检索与文本挖掘的常用加权技术。TF-IDF 是一种统计方法，用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度。

TF-IDF 由 TF（Term frequency，词频）和 IDF（Inverse document frequency，逆文档频率）两部分组成。计算公式为：
$$
tf_{ij} = \frac{n_{ij}}{\sum_{k}n_{kj}}
$$
式中，分子 𝑛𝑖𝑗 表示词 𝑖 在文档 𝑗 中出现的频次。分母则是所有词频次的总和，也就是所有词的个数。
$$
idf_{i} = log\left ( \frac{\left | D \right |}{1+\left | D_{i} \right |} \right ) 
$$
式中，|𝐷| 代表文档的总数，分母部分 |𝐷𝑖| 则是代表文档集中含有 𝑖 词的文档数。原始公式是分母没有 +1 的，这里 +1 是采用了**拉普拉斯平滑**，避免了有部分新的词没有在语料库中出现而导致分母为零的情况出现。

最后，把 TF 和 IDF 两个值相乘就可以得到 TF-IDF 的值。即：
$$
tf*idf(i,j)=tf_{ij}*idf_{i}= \frac{n_{ij}}{\sum_{k}n_{kj}} *log\left ( \frac{\left | D \right |}{1+\left | D_{i} \right |} \right )
$$
同样，scikit-learn 提供了 `sklearn.feature_extraction.text.TfidfVectorizer` 可用于 TF-IDF 转换。

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(corpus)
tfidf.toarray()
```

```
['The elephant sneeze at the sight of potato.', 'Bat can see via echolocation. See the bat sight sneeze!', 'Wonder, she open the door to the studio.']
```

Out[20]:

```
array([[0.39066946, 0.        , 0.        , 0.        , 0.        ,
        0.39066946, 0.39066946, 0.        , 0.39066946, 0.        ,
        0.        , 0.29711419, 0.29711419, 0.        , 0.46147135,
        0.        , 0.        , 0.        ],
       [0.        , 0.56555816, 0.28277908, 0.        , 0.28277908,
        0.        , 0.        , 0.        , 0.        , 0.56555816,
        0.        , 0.21506078, 0.21506078, 0.        , 0.16701388,
        0.        , 0.28277908, 0.        ],
       [0.        , 0.        , 0.        , 0.36772387, 0.        ,
        0.        , 0.        , 0.36772387, 0.        , 0.        ,
        0.36772387, 0.        , 0.        , 0.36772387, 0.43436728,
        0.36772387, 0.        , 0.36772387]])
```

## Word2Vec模型

[ *Word2Vec 模型*](https://doi.org/10.1017%2FCBO9781139058452.002) 是 Google 团队于 2015 年提出来的一种字词的向量表示法，又被称为「词嵌入」。

无论是词袋模型，还是 TF-IDF 模型，它们均是使用**离散化的向量值**来表示文本。这些编码是**任意**的，并未提供有关字词之间可能存在的**相关性**。将字词表示为唯一的离散值还会导致**数据稀疏性**，并且通常意味着我们可能需要更多数据才能成功训练统计模型。

[ *向量空间模型*](https://en.wikipedia.org/wiki/Vector_space_model) 在连续向量空间中表示（嵌入）字词，其中语义相似的字词会映射到附近的点（在**彼此附近嵌入**）。向量空间模型在 NLP 方面有着悠久而丰富的历史，但所有方法均以某种方式依赖于分布假设，这种假设指明在相同上下文中显示的字词语义相同。

它分为两种类型：**连续词袋模型**（CBOW）和 **Skip-Gram** 模型。从算法上看，两种模型比较相似，只是 CBOW 从源**上下文**字词（the cat sits on the）中**预测目标**字词（例如 mat），而 Skip-Gram 则逆向而行，从**目标字词**中**预测源上下文**字词。

Word2Vec 词嵌入过程一般常用 [ *Gensim*](https://github.com/RaRe-Technologies/gensim) 库来处理。

其提供了封装好的高效 Word2Vec 处理类 `gensim.modelsWord2Vec()`。其常用参数有：

```
- size: 词嵌入的维数，表示每个单词嵌入后的向量长度。
- window: 目标字与目标字周围的字之间的最大距离。
- min_count: 训练模型时要考虑的最小字数，出现小于此计数的单词将被忽略。
- sg: 训练算法，CBOW(0)或 Skip-Gram(1)。
```

- 代码

```python
# 使用 Word2Vec 对前面的示例文本进行词嵌入操作。
from gensim.models import Word2Vec
import warnings
warnings.filterwarnings('ignore')

# 分词之后的示例文本
sentences = [['the', 'elephant', 'sneeze', 'at', 'the', 'sight', 'of', 'potato'],
             ['bat', 'can', 'see', 'via', 'echolocation', 'see', 'the', 'bat', 'sight', 'sneeze'],
             ['wonder', 'she', 'open', 'the', 'door', 'to', 'the', 'studio']]

# 训练模型
model = Word2Vec(sentences, size=20, min_count=1)
# 输出该语料库独立不重复词
print(list(model.wv.vocab))
# 输出 elephant 单词词嵌入后的向量
model['elephant']

'''
把 Word2Vec 嵌入后的词在空间中绘制处理。由于上面模型设置了 size=20，所以需要使用 PCA 降维把嵌入后的向量降维为二维才能够在平面中可视化出来。
'''
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
%matplotlib inline

# PCA 降维
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)

# 绘制散点图，并将单词标记出来
plt.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))
```

- Google预训练模型

Google 在 [ *Word2Vec Project*](https://code.google.com/archive/p/word2vec/) 上发布了一个预先训练过的 Word2Vec 模型。该模型使用谷歌新闻数据（约 1000 亿字）进行训练，其包含了 300 万个单词和短语，并且使用 300 维词向量表示。由于该预训练词嵌入模型大小为 3.4 GB，这里就不再演示了。你可以在 [ *本地下载*](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing) 下来，并通过以下代码加载模型。

```python
from gensim.models import KeyedVectors

filename = 'GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True)
```

当得到每个单词的词嵌入向量后，就可以通过**直接求和**或者其他**加权求和**方法得到**一段文本的向量特征**，从而可以**传入分类器**进行训练。