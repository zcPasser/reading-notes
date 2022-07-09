[TOC]

# NLP

## 基础入门

### 字符串基础操作及应用

#### NLP介绍

- 定义

```
利用计算机的强大的运算能力，采用统计手段来对语言进行处理，然后获得需要的信息，以达到最终想要的目的，而使用各种方法的一门技术。
```

- 分类

```
1、= 2个部分 = NLU（自然语言理解） + NLG（自然语言生成）。
2、NLU构成= 词义分析 + 句法分析 + 语义分析
3、NLG构成= 文本规划 + 语句规划 + 实现
从结构化的数据（可以通俗理解为自然语言理解分析后的数据）以读取的方式自动生成文本。

文本规划：完成结构化数据中的基础内容规划。
语句规划：从结构化数据中组合语句来表达信息流。
实现：产生语法通顺的语句来表达文本。
```

- 研究与应用

```
信息检索：对大规模文档进行索引。
语音识别：识别包含口语在内的自然语言的声学信号转换成符合预期的信号。
机器翻译：将一种语言翻译成另外一种语言。
智能问答：自动回答问题。
对话系统：通过多回合对话，跟用户进行聊天、回答、完成某项任务。
文本分类：将文本自动归类。
情感分析：判断某段文本的情感倾向
文本生成：根据需求自动生成文本
自动文摘：归纳，总结文本的摘要。
```

- 相关术语

```
分词：词是 NLP 中能够独立活动的有意义的语言成分。即使某个中文单字也有活动的意义，但其实这些单字也是词，属于单字成词。

词性标注：给每个词语的词性进行标注，比如 ：跑/动词、美丽的/形容词等等。

命名实体识别：从文本中识别出具有特定类别的实体。像是识别文本中的日期，地名等等。

词义消歧：多义词判断最合理的词义。

句法分析：解析句子中各个成分的依赖关系。

指代消解：消除和解释代词「这个，他，你」等的指代问题。
```

#### 字符串操作

```python
# 统计子串出现的次数
1、.count() 方法返回特定的子串在字符串中出现的次数。
seq = '12345,1234,123,12,1'
seq1 = '1'
a = seq.count(seq1)

# 去除字符串
1、.strip()方法可以去除字符串首尾的指定符号。无指定时，默认去除空格符 ' ' 和换行符 '\n'。
2、使用 .lstrip() 方法。
3、使用.rstrip() 方法来单独去除末尾的字符

# 拼接字符串
1、用运算符 + 来简单暴力的拼接
2、用特定的符号拼接起来的字符的时候，可以用 .join() 方法来进行拼接
seq = ['2018', '10', '31']
seq = '-'.join(seq)  # 用 '-' 拼接， join中存放迭代器对象
seq

# 比较
1、加载 operator 工具，它是 Python 的标准库。直接通过 import 调用即可。operator 从左到右第一个字符开始，根据设定的规则比较，返回布尔值（ True，False ）。
2、直接使用运算符比较，a < b:

# 字符串大小写转换
1、.upper() 和 .lower() 可以很方便的完成这个任务。
seq = 'appLE'
seq = seq.upper()

# 翻转字符串
1、切片方式
seq = '12345'
seq = seq[::-1]

# 查找字符串
1、.find() 方法，在序列当中找到子串的起始位置。PS：第一个位置是 0 。
字符串中不存在子串返回 -1。
2、.index()方法如果没有找到子串，会报错提醒：substring not found。

# 字符串切分
1、序列截取
s[start: stop[: step]]
2、一个字符串按照某个字符切分开处理
split()函数可以完成这个操作,函数返回一个由切分好的字符串组成的列表。
seq = '今天天气很好，我们出去玩'
seq = seq.split('，')

# 判断子串是否存在
1、用 in 来作出判断。
in 关键字可以用在任何容器对象上，判断一个子对象是否存在于容器当中，并不局限于判断字符串是否存在某子串，还可以用在其他容器对象例如 list，tuple，set 等类型。

# 字符串代替
1、可以用到 .replace(a,b) ，他可以将某字符串中的 a 字符串 替换成 b 字符串。
seq = '2018-11-11'
seq = seq.replace('-', '/')

# 检查字符串
1、.startswish() 方法
2、用 .endswith() 来确定字符串是否以某段字符串结尾
3、检查字符串是否由纯数字构成。
seq = 's123'
seq.isdigit()
```

#### python中的正则

```python
# 1、首先，我们需要通过 re.compile() 将编写好的正则表达式编译为一个实例。

# 2、然后我们对字符串进行匹配，并对匹配结果进行操作。

# 3、re.search() 方法。 这个方法是将正则表达式与字符串进行匹配，如果找到第一个符合正则表达式的结果，就会返回，然后匹配结果存入group()中供后续操作，匹配失败，返回None。该方法从开始处扫描整个字符串，匹配1次。

# 4、.findall()：这个方法可以找到符合正则表达式的所有匹配结果。
# 使用了 \d 规则的正则表达式，这个正则表达式可以替我们识别数字。
pattern = re.compile(r'\d')
pattern.findall('o1n2m3k4')
#  \D 正则表达式，这个可以匹配一个非数字字符
pattern = re.compile('\D')
pattern.findall('1A2B3C4D')

# 5、.match() 方法与 .search() 方法类似，只匹配一次，并且只从字符串的开头开始匹配。同样，match 结果也是存在 group() 当中。
# 下面我们试着用 .match() 来匹配字符串的末尾字符，发现代码报错。这是因为 .match() 只从开头匹配。若匹配不成功，则 group()不会有内容。
pattern = re.compile('c')
pattern.match('comcdc').group()
# 失败
pattern = re.compile('1')
pattern.match('abcdefg1').group()

# 加载 re 模块
import re

# 将正则表达式编写成实例
pattern = re.compile(r'[0-9]{4}')
times = ('2018/01/01', '01/01/2019', '01.2017.01')

for time in times:
    match = pattern.search(time)
    if match:
        print('年份有：', match.group())
```



### 中英文分词方法及实现

#### 英文分词

- `.split()` 方法

下面来实现对多个英文文本分词，要求同时以 `,` ， `.` ， `?` ， `!` ， `  ` 五个符号分词。

```python
# 为了方便调用，我们将代码写成一个函数。首先对原文本以其中一个规则切分后，再对分好后的文本进行下一个规则的切分，再对分好的文本进行切分，直到按 5 个规则切分完成，最后将切分好的词添加进 tokenized_text 并返回。
def tokenize_english_text(text):
    # 首先，我们按照标点来分句
    # 先建立一个空集用来，用来将分好的词添加到里面作为函数的返回值
    tokenized_text = []
    # 一个 text 中可能不止一个内容，我们对每个文本单独处理并存放在各自的分词结果中。
    for data in text:
        # 建立一个空集来存储每一个文本自己的分词结果，每对 data 一次操作我们都归零这个集合
        tokenized_data = []
        # 以 '.'分割整个句子，对分割后的每一小快 s：
        for s in data.split('.'):
            # 将's'以 '？'分割，分割后的每一小快 s2：
            for s1 in s.split('?'):
                # 同样的道理分割 s2，
                for s2 in s1.split('!'):
                    # 同理
                    for s3 in s2.split(','):
                        # 将 s3 以空格分割，然后将结果添加到 tokenized_data 当中
                        tokenized_data.extend(
                            s4 for s4 in s3.split(' ') if s4 != '')
                        # 括号内的部分拆开理解
                        # for s4 in s3.split(' '):
                        #    if s4!='':  这一步是去除空字符''。注意与' ' 的区别。
        # 将每个 tokenized_data 分别添加到 tokenized_text 当中
        tokenized_text.append(tokenized_data)

    return tokenized_text
```



#### 中文分词

##### 困难

- 歧义

- 分词界限

##### 方法

###### 机械分词方法

```
又叫做基于规则的分词方法：这种分词方法按照一定的规则将待处理的字符串与一个词表词典中的词进行逐一匹配，若在词典中找到某个字符串，则切分，否则不切分。机械分词方法按照匹配规则的方式，又可以分为：正向最大匹配法，逆向最大匹配法和双向匹配法三种。
```

- 正向最大匹配法

```
指从左向右按最大原则与词典里面的词进行匹配。假设词典中最长词是  𝑚  个字，那么从待切分文本的最左边取  𝑚  个字符与词典进行匹配，如果匹配成功，则分词。如果匹配不成功，那么取  𝑚−1  个字符与词典匹配，一直取直到成功匹配为止。
```

```
句子：中华民族从此站起来了
词典："中华"，"民族"，"从此"，"站起来了"

使用 MM 法分词：

第一步：词典中最长是 4 个字，所以我们将 【中华民族】 取出来与词典进行匹配，匹配失败。
第二步：于是，去掉 【族】，以 【中华民】 进行匹配，匹配失败
第三步：去掉 【中华民】 中的 【民】，以 【中华】 进行匹配，匹配成功。
第四步：在带切分句子中去掉匹配成功的词，待切分句子变成 【民族从此站起来了】。
第五步：重复上面的第 1 - 4 步骤
第六步：若最后一个词语匹配成功，结束。
```

-- 流图

![](D:\事务\我的事务\拓展学习\笔记\pictures\正向最大匹配算法.png)

```
1、导入分词词典 dic，待分词文本 text，创建空集 words 。
2、遍历分词词典，找到最长词的长度，max_len_word 。
3、将待分词文本从左向右取 max_len=max_len_word 个字符作为待匹配字符串 word 。
4、将 word 与词典 dic 匹配
5、若匹配失败，则 max_len = max_len - 1 ，然后
6、重复 3 - 4 步骤
7、匹配成功，将 word 添加进 words 当中。
8、去掉待分词文本前 max_len 个字符
9、重置 max_len 值为 max_len_word
10、重复 3 - 8 步骤
11、返回列表 words
```

-- 案例

```python
# 文本
text = '我们是共产主义的接班人'

# 词典
dic = ('我们', '是', '共产主义', '的', '接班', '人', '你', '我', '社会', '主义')

# 最长词长度为
max_len_word0 = max([len(key) for key in dic ])

sent = text
words = []   # 建立一个空数组来存放分词结果：
max_len_word = max_len_word0
# 判断 text 的长度是否大于 0，如果大于 0 则进行下面的循环
while len(sent) > 0:
    # 初始化想要取的字符串长度
    # 按照最长词长度初始化
    word_len = max_len_word
    # 对每个字符串可能会有(max_len_word)次循环
    for i in range(0, max_len_word):
        # 令 word 等于 text 的前 word_len 个字符
        word = sent[0:word_len]
        # 为了便于观察过程，我们打印一下当前分割结果
        print('用 【', word, '】 进行匹配')
        # 判断 word 是否在词典 dic 当中
        # 如果不在词典当中
        if word not in dic:
            # 则以 word_len - 1
            word_len -= 1
            # 清空 word
            word = []
        # 如果 word 在词典当中
        else:
            # 更新 text 串起始位置
            sent = sent[word_len:]
            # 为了方便观察过程，我们打印一下当前结果
            print('【{}】 匹配成功，添加进 words 当中'.format(word))
            print('-'*50)
            # 把匹配成功的word添加进上面创建好的words当中
            words.append(word)
            # 清空word
            word = []
```

- 逆向最大匹配法

```
原理与正向法基本相同，唯一不同的就是切分的方向与 MM 法相反。逆向法从文本末端开始匹配，每次用末端的最长词长度个字符进行匹配。
```

```
由于汉语言结构的问题，里面有许多偏正短语。
因此，如果采用逆向匹配法，可以适当提高一些精确度。
```

- 双向匹配法

```
双向最大匹配法（Bi-direction Matching Method ，BMM）则是将正向匹配法得到的分词结果与逆向匹配法得到的分词结果进行比较，然后按照最大匹配原则，选取次数切分最少的作为结果。
```

###### 统计分词方法

基于的字典的方法实现比较简单，而且性能也还不错。但是其有一个缺点，那就是不能切分 [<i class="fa fa-external-link-square" aria-hidden="true"> 未登录词</i>](https://baike.baidu.com/item/%E6%9C%AA%E7%99%BB%E5%BD%95%E8%AF%8D/10993635?fr=aladdin) ，也就是不能切分字典里面没有的词。

- 语料统计方法

```
对于语料统计方法可以这样理解：我们已经有一个由很多个文本组成的的语料库 D ，假设现在对一个句子【我有一个苹果】进行分词。其中两个相连的字 【苹】【果】在不同的文本中连续出现的次数越多，就说明这两个相连字很可能构成一个词【苹果】。与此同时 【个】【苹】 这两个相连的词在别的文本中连续出现的次数很少，就说明这两个相连的字不太可能构成一个词【个苹】。所以，我们就可以利用这个统计规则来反应字与字成词的可信度。当字连续组合的概率高过一个临界值时，就认为该组合构成了一个词语。
```

- 序列标注方法

```
序列标注方法则将中文分词看做是一个序列标注问题。首先，规定每个字在一个词语当中有着 4 个不同的位置，词首 B，词中 M，词尾 E，单字成词 S。我们通过给一句话中的每个字标记上述的属性，最后通过标注来确定分词结果。
```

```
例如：我今天要去实验室

标注后得到：我/S 今/B 天/E 要/S 去/S 实/B 验/M 室/E

标注序列是：S  B  E  S  S  B  M  E

找到 S 、B 、 E 进行切词：S / B E / S / S / B M E /

所以得到的切词结果是：我 / 今天 / 要 / 去 / 实验室

在训练时，输入中文句子和对应的标注序列，训练完成得到一个模型。在测试时，输入中文句子，通过模型运算得到一个标注序列。然后通过标注序列来进行切分句子。
```

-- jieba分词

```python
# Python 用第三方的中文分词工具jieba 工具，用的是隐马尔可夫模型与字典相结合的方法，比直接单独使用隐马尔可夫模型来分词效率高很多，准确率也高很多。

# jieba有3种模式
# 全模式
# 精确模式
# 搜索引擎模式

# 全模式
# jieba 是将分词后的结果存放在生成器当中的。
# 无法直接显示，若想要显示，可以下面这样。用 ‘|’ 把生成器中的词串起来显示。这个方法在下面提到的精确模式和搜索引擎模式中同样适用。
string = '我来到北京清华大学'
seg_list = jieba.cut(string, cut_all=True)
'| '.join(seg_list)
# '我| 来到| 北京| 清华| 清华大学| 华大| 大学'

# 精确模式
seg_list = jieba.cut(string, cut_all=False)
'|'.join(seg_list)
# '我|来到|北京|清华大学'

# 搜索引擎模式
seg_list = jieba.cut_for_search(string)
'|'.join(seg_list)
# '我|来到|北京|清华|华大|大学|清华大学'

# 可以看到，全模式和搜索引擎模式，jieba 会把全部可能组成的词都打印出来。在一般的任务当中，我们使用默认的精确模式就行了，在模糊匹配时，则需要用到全模式或者搜索引擎模式。

# jieba 在某些特定的情况下分词，可能表现不是很好。比如一篇非常专业的医学论文，含有一些特定领域的专有名词。不过，为了解决此类问题， jieba 允许用户自己添加该领域的自定义词典，我们可以提前把这些词加进自定义词典当中，来增加分词的效果。调用的方法是：jieba.load_userdic()。
# 自定义词典的格式要求每一行一个词，有三个部分，词语，词频（词语出现的频率），词性（名词，动词……）。其中，词频和词性可省略。用户自定义词典可以直接用记事本创立即可，但是需要以 utf-8 编码模式保存。 格式像下面这样：
#   凶许 1 a
#   脑斧 2 b
#   福蝶 c
#   小局 4 
#   海疼
 
# 动态修改词典
# 使用 add_word(word, freq=None, tag=None) 和 del_word(word) 可在程序中动态修改词典。
# 使用 suggest_freq(segment, tune=True) 可调节单个词语的词频，使其能（或不能）被分出来。

# 举例：添加新词
jieba.suggest_freq(('今天', '天气'), True)
seg_list = jieba.cut(string, cut_all=False)
'|'.join(seg_list)
# 修改后：'今天|天气|不错'
# 修改前：'今天天气|不错'
# 或者 从词典直接删除该短语
jieba.del_word('今天天气')
seg_list = jieba.cut(string, cut_all=False)
'|'.join(seg_list)

# 举例：修改词频
```

```python
# 过滤器——过滤如停用词等对象
# 停用词表
stopwords = ('的', '地', '得')
string = '我喜欢的和你讨厌地以及最不想要得'
# 首先创建一个空数组来存放过滤后的词语，然后通过循环迭代的方法，将过滤后的词语依次添加到刚刚建立的空数组当中。
a = []
seg_list = jieba.cut(string, cut_all=False)
for word in seg_list:
    if word not in stopwords:
        a.append(word)
```

### 中文邮件文本分类实战

```
文本分类是很多其他任务的基本型，本次实验将会介绍如何将文本数据转化为数值型的特征数据。同时，讲解机器学习当中的支持向量机算法，并且用 Python 实现一个对 10001 个邮件样本进行分类的任务。
```

#### 文本分类简介

##### 定义

一般而言，文本分类是指在**一定的规则**下，**根据内容自动确定文本类别**这一过程。

##### 类别

- 二分类

也是最基础的分类，顾名思义是将文本归为两种类别，比如将正常邮件邮件划分问题，垃圾邮件或者正常邮件。一段影评，判断是好评还是差评的问题。

- 多分类

是将文本划分为多个类别，比如将新闻归为政治类，娱乐类，生活类等等。

- 多标签分类

是给文本贴上多个不同的标签，比如一部小说可以同时被划分为多个主题，可能既是修仙小说，又是玄幻小说。

##### 方法

- 传统机器学习文本分类算法

**特征提取 + 分类器**。就是将文本转换成**固定维度的向量**，然后送到**分类器**中进行分类。

- 深度学习文本分类算法

可以**自动提取特征**，实现**端到端的训练**，有较强的特征表征能力，所以深度学习进行文本分类的效果往往要好于传统的方法。

#### 支持向量机SVM

##### 原理

SVM 作为**传统机器学习**的一个非常重要的分类算法，**给定训练样本**，支持向量机找到一个**划分超平面**，将不同的**类别划分**开来。通俗来讲，这样的超平面有很多，支持向量机就是要找到位于两类训练样本「正中间」的划分超平面。

##### 使用方法

#### 中文邮件分类

##### 实验步骤总述

1. 导入数据，并进行分词和剔除停用词。
2. 划分训练集和测试集。
3. 将文本数据转化为数字特征数据。
4. 构建分类器。
5. 训练分类器。
6. 测试分类器。

##### 数据准备

```python
# 本次用到的数据包含 3 个文件， ham_data.txt 文件里面包含 5000 条正常邮件样本，spam_data.txt 文件里面包含 5001 个垃圾邮件样本，stopwords 是停用词表。整个数据集放到了蓝桥云课的服务器上提供下载。
!wget - nc "http://labfile.oss.aliyuncs.com/courses/1208/ham_data.txt"
!wget - nc "http://labfile.oss.aliyuncs.com/courses/1208/spam_data.txt"
!wget - nc "http://labfile.oss.aliyuncs.com/courses/1208/stop_word.txt"

# 获得了样本之后，首先要做是给正常邮件和垃圾邮件贴上标签，我们用 1 代表正常邮件，0 代表垃圾邮件。
path1 = 'ham_data.txt'  # 正常邮件存放地址
path2 = 'spam_data.txt'  # 垃圾邮件地址

# 下面用 utf-8 编码模式打开正常样本：
h = open(path1, encoding='utf-8')

# 准备的数据是每一行一封邮件，这里我们要用 readlines() 来以行来读取文本的内容。
h_data = h.readlines()
h_data[0:3]  # 显示前3封正常邮件

s = open(path2, encoding='utf-8')
s_data = s.readlines()
s_data[0:3]  # 显示前3个封垃

# 读取之后，我们的 h_data 数是由 5000 条邮件字符串组成的正常邮件样本集， s_data 是由 5001 条邮件字符串组成的垃圾邮件样本集。下面我们为将两个样本组合起来，并贴上标签，将正常邮件的标签设置为 1，垃圾邮件的标签设置为 0。
# 生成一个 len(h_data) 长的的一维全 1 列表：
import numpy as np

h_labels = np.ones(len(h_data)).tolist()  # 生成一个len(h_data)长的的一维全1列表
h_labels[0:10]  # 我们显示前10个数据
# 生成一个 len(s_data) 长的的一维全 0 列表：
s_labels = np.zeros(len(s_data)).tolist()
s_labels[0:10]  # 我们显示前10个数据

# 拼接样本集和标签集：
datas = h_data + s_data  # 将正常样本和垃圾样本整合到datas当中
labels = h_labels + s_labels

# 因为我们没有事先准备测试集，所以我们在 10001 个样本当中，随机划出 25% 个样本和标签来作为我们的测试集，剩下的 75% 作为训练集来进行训练我们的分类器。这里我们可以用到 scikit-learn 工具里面的 `train_test_split` 类。
sklearn.model_selection.train_test_split(datas, labels, test_size=0.25, random_state=5 )
# datas : 样本集
# labels: 标签集
# train_test_split:划分到测试集的比例
# random_state:随机种子，取同一个的随机种子那么每次划分出的测试集是一样的。
from sklearn.model_selection import train_test_split

train_d, test_d, train_y, test_y = train_test_split(
    datas, labels, test_size=0.25, random_state=5)
# 调用这个函数之后，就可以得到：
# train_d:样本集
# test_d:测试集
# train_y:样本标签
# test_y:测试标签

```

##### 分词

```python
# 将分词设计成 tokenize_words 函数，供后续直接调用。
import jieba

def tokenize_words(corpus):
    tokenized_words = jieba.cut(corpus)
    tokenized_words = [token.strip() for token in tokenized_words]
    return tokenized_words
```

#####  去除停用词

```python
def remove_stopwords(corpus):  # 函数输入为样本集
    sw = open('stop_word.txt', encoding='utf-8')  # stopwords 停词表
    sw_list = [l.strip() for l in sw]  # 去掉文本中的回车符，然后存放到 sw_list 当中
    # 调用前面定义好的分词函数返回到 tokenized_data 当中
    tokenized_data = tokenize_words(corpus)
    # 过滤停用词，对每个在 tokenized_data 中的词 data 进行判断
    # 如果 data 不在 sw_list 则添加到 filtered_data 当中
    filtered_data = [data for data in tokenized_data if data not in sw_list]
    # 用''将 filtered_data 串起来赋值给 filtered_datas
    filtered_datas = ' '.join(filtered_data)
    return filtered_datas  # 返回去停用词之后的 datas

# 接下来，构建一个函数完成分词和剔除停用词。这里使用 tqdm 模块显示进度。
from tqdm.notebook import tqdm

def preprocessing_datas(datas):
    preprocessed_datas = []
    # 对 datas 当中的每一个 data 进行去停用词操作
    # 并添加到上面刚刚建立的 preprocessed_datas 当中
    for data in tqdm(datas):
        data = remove_stopwords(data)
        preprocessed_datas.append(data)

    return preprocessed_datas  # 返回去停用词之后的新的样本集
# 然后用上面预处理函数对样本集进行处理。
pred_train_d = preprocessing_datas(train_d)
pred_train_d[0]
# 同样，对测试集进行预处理：
pred_test_d = preprocessing_datas(test_d)
pred_test_d[0]
# 通过上面两步，我们得到了分词过后并且去除停用词了的样本集 pred_train_d 和 测试集 pred_test_d。
```

##### 特征提取

```
在进行分词及去停用词处理过后，得到的是一个分词后的文本。现在我们的分类器是 SVM，而 SVM 的输入要求是数值型的特征。这意味着我们要将前面所进行预处理的文本数据进一步处理，将其转换为数值型数据。转换的方法有很多种，为了便于理解，这里使用 TF-IDF 方法。为了更好的理解 TF-IDF，我们先从词袋模型开始讲解。
```

###### 词袋模型

```
词袋模型是最原始的一类特征集，忽略掉了文本的语法和语序，用一组无序的单词序列来表达一段文字或者一个文档。可以这样理解，把整个文档集的所有出现的词都丢进袋子里面，然后无序的排出来（去掉重复的）。对每一个文档，按照词语出现的次数来表示文档。
```

- 举例

![](D:\事务\我的事务\拓展学习\笔记\pictures\词袋模型举例.jpg)

###### TF-IDF模型

```
这种模型主要是用词汇的统计特征来作为特征集。TF-IDF 由两部分组成：TF（Term frequency，词频），IDF（Inverse document frequency，逆文档频率）两部分组成。
```

- TF词频

![](D:\事务\我的事务\拓展学习\笔记\pictures\TF词频.jpg)

- IDF逆文档频率

![](D:\事务\我的事务\拓展学习\笔记\pictures\IDF.jpg)

将每个句子中的每个词的IF-IDF值添加到向量表示出来就是每个句子TF-IDF特征

```python

# python实现
# 在 Python 当中，我们可以通过 scikit-learn 来实现 TF-IDF 模型。并且，使用 scikit-learn 库将会非常简单。这里主要用到了 TfidfVectorizer() 类。

# sklearn.feature_extraction.text.TfidfVectorizer(min_df=1,norm='l2',smooth_idf=True,use_idf=True,ngram_range=(1,1)）

# min_df： 忽略掉词频严格低于定阈值的词。
# norm ：标准化词条向量所用的规范。
# smooth_idf：添加一个平滑 IDF 权重，即 IDF 的分母是否使用平滑，防止 0 权重的出现。
# use_idf： 启用 IDF 逆文档频率重新加权。
# ngram_range：同词袋模型

# 首先加载 TfidfVectorizer 类，并定义 TF-IDF 模型训练器 vectorizer 。
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    min_df=1, norm='l2', smooth_idf=True, use_idf=True, ngram_range=(1, 1))

# 对预处理过后的 pred_train_d 进行特征提取：
tfidf_train_features = vectorizer.fit_transform(pred_train_d)
# 通过这一步，我们得到了 7500 个 28335 维数的向量作为我们的训练特征集。我们可以查看转换结果，这里为了便于观察，使用 toarray 方法转换成为数组数据。
tfidf_train_features.toarray()[0]
# 用训练集训练好特征后的 vectorizer 来提取测试集的特征： 注意这里不能用 vectorizer.fit_transform() 要用 vectorizer.transform()，否则，将会对测试集单独训练 TF-IDF 模型，而不是在训练集的词数量基础上做训练。这样词总量跟训练集不一样多，排序也不一样，将会导致维数不同，最终无法完成测试。
tfidf_test_features = vectorizer.transform(pred_test_d)
tfidf_test_features
```

##### 分类

```python
# 在获得 TF-IDF 特征之后，我们可以调用 SGDClassifier() 类来训练 SVM 分类器。
# sklearn.linear_model.SGDClassifier(loss='hinge')
# SGDClassifier 是一个多个分类器的组合，当参数 loss='hinge' 时是一个支持向量机分类器。

# 加载 SVM 分类器，并调整 loss = 'hinge'。
from sklearn.linear_model import SGDClassifier

svm = SGDClassifier(loss='hinge')
# 然后我们将之前准备好的样本集和样本标签送进 SVM 分类器进行训练。
svm.fit(tfidf_train_features, train_y)
# 接下来我们用测试集来测试一下分类器的效果。
predictions = svm.predict(tfidf_test_features)

# 为了直观显示分类的结果，我们用 scikit-learn 库中的 accuracy_score 函数来计算一下分类器的准确率 。
# sklearn.metrics.accuracy_score(test_l, prediction)
# 这个函数的作用是为了计算 test_l 中与 prediction 相同的比例。即准确率。

# 用测试标签和预测结果 计算分类准确率。np.round(X,2) 的作用是 X 四舍五入后保留小数点后 2 位数字。
from sklearn import metrics

accuracy_score = np.round(metrics.accuracy_score(test_y, predictions), 2)
accuracy_score
```

