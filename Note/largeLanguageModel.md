## LLM List

|     名称     |   研发机构   |  架构   |                           关键词                            |
| :----------: | :----------: | :-----: | :---------------------------------------------------------: |
|   GPT-3.5    |    OpenAI    | Decoder |                           General                           |
|    GPT-4     |    OpenAI    | Decoder |                           General                           |
|   Llama-2    |     Meta     | Decoder | Rejection Sampling, grouped-query attention, iterative RLHF |
|    Falcon    |              |         |                                                             |
|    Vicuna    |              |         |                                                             |
|    Alpaca    |              |         |                                                             |
|    Gemini    |    Google    | Decoder |                                                             |
|    Claude    |  Anthropic   |         |                                                             |
| BloombergGPT |  Bloomberg   |         |                                                             |
|  Character   | character ai |         |                                                             |
| ColossalChat | Colossal-AI  |         |                                                             |
|    Bloom     |              | Decoder |                                                             |

* [GPT-4]()
* [chatGLM]()
  * ![image-20240601181302132](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240601181302132.png)不同于decoder-only、encoder-only和en-de任务，提出了自回归空格填充任务autoregressive blank infilling：partA是双向，partB是单项
  * 二维位置编码：partA（未mask部分）和partB（mask部分）
  * **chatGLM2**又改回decoder-only，加上flashAttention、MQA、RoPE
* [Llama-2](https://arxiv.org/pdf/2307.09288.pdf)
  * RMS初始化
    * RMS均方根标准化，layerNorm在单个样本内进行
    * ![image-20240530204803225](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240530204803225.png)
    * ![image-20240530204929022](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240530204929022.png)
    * RMS不计算均值![image-20240530205023094](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240530205023094.png)
    * 训练更稳定、收敛更快；norm一般在激活函数之前，让激活函数的输入更稳定
  * swiGLU
    * swish激活（比ReLU更平滑，处处可微，relu非负区域不可微） + GLU门控（线性变换后接sigmoid）
  * FFN层
    * 两个mlp和一个swiGLU
    * 目的是提升表达能力，一方面通过两个mlp升维再降维，另一方面是非线性激活函数
  * RoPE
  * GQA
  * 只收集了27450条SFT数据，发现SFT数据的质量更重要，少量高质量SFT数据就可以表现得不错
  * 两个模型分别采样两条输出，标注团队标注哪个更好；评价指标包括Helpful和Safety【
  * 标注偏好数据时，加入了better的程度，并统一到了ranking loss里
  * RLHF之后，Actor的输出会慢慢发生偏移，所以reward model的效果就会越来越差，所以有必要不断收集新的偏好数据来更新reward model
  * 用了两个reward model，分别是helpful reward和safety reward
* CodeLlama
* Llama-3
  * 8K上下文，128K词表（llama1-32k），15T的token（增加代码数据）
  * 增强训练稳定性，采用RMSNorm替换layerNorm
  * 提高模型性能，采用SwiGLU作为激活函数
  * RoPE，GQA（8B和70B）
  * sft、拒绝采样、近端策略优化ppo、直接策略优化dpo
* [Falcon](https://arxiv.linfen3.top/pdf/2306.01116.pdf)
  * 在数据上做了详细介绍，强调了过滤、去重的重要性，即使在公开网页数据上也能够训练出足够强大的模型
  * 一些数据清洗的方法：语言识别、规则过滤、基于ML的文档去重、去重
* [Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/)
  * 70K微调数据，基于Llama进行微调
* [Alpaca]()
  * 50K微调数据（来自GPT-3的self-instruct），基于Llama进行微调
* [Gemini](https://storage.googleapis.com/deepmind-media/gemini/gemini_1_report.pdf)
  * State-of-art multimodel模型，有很惊艳的结果，比如给它一副学生做的物理作业题，它可以纠正学生的错误，并给出正确的解题步骤
  * 视觉上的工作基于前面的Flamingo, Coca
  * 丰富的输入，支持文本、语音、图片和视频
  * 能直接写文献综述
  * 原生多模态
* [BloombergGPT](https://arxiv.dosf.top/pdf/2303.17564.pdf?utm_source=substack&utm_medium=email)
  * 在领域内、通用数据上训练的，比例接近1：1
  * Tokenizer是Unigram模型，关于Unigram:Unigram从一个大的词典开始train，每次是移除token，而不是融合，直到词典的大小满足要求.对词典内的每一个符号，算法都会算出移除这个符号总的损失会升高多少，找到升高值比较小的那些符号，然后将它们移除。也就是不太需要的符号可以被移除掉。因损失升高的很小。
* [Character]()
  * 很有意思的模型，用户可以自己创建人物，可以跟人物对话，甚至可以拉群，让角色之间互相闹腾
  * 有社区，相当多的活跃人数



# 继续预训练

有新数据时，有两种训练模型的方式：

1. 继续预训练：采用1)中预训练的模型，并在数据集D2上继续预训练
2. 在合并数据集上重新训练：像1)中一样使用随机权重初始化模型，但在数据集D1和D2的合并上进行训练

方式1会造成灾难性遗忘，且很难找到良好的学习率调度，可以参考的技巧：

1. 重新热身和重新衰减学习率（基本上，重新应用典型的学习率调度）
2. 将原始预训练数据（D1）的一小部分（例如5%）添加到新数据集（D2）中，以防止灾难性遗忘

[Simple and Scalable Strategies to Continually Pre-train Large Language Models](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2403.08763)



# 提高大模型推理能力

《Large Language Models are Zero-Shot Reasoners》

**zero-shot-cot：**生成之前拼接**"Let's think step by step."**



《Self-Consistency Improves Chain of Thought Reasoning in Language Models》

随机采样一波cot结构，投票一波看看



《Automatic Chain of Thought Prompting in Large Language Models》

从一个问题库中通过向量搜索相似的问题，用zero shot cot的方式生成例子，称为retrieval-q-cot，还有一种对比就是随机抽，称为random-q-cot



《Thread of Thought Unraveling Chaotic Contexts》

模仿人在面对复杂信息时使用的认知策略，将其分解为可消化的片段，提取关键点，并持续专注地浏览材料。这种增量方法促进了更结构化和更连贯的推理，被证明在混乱环境中特别有利。

Let's think step by step. ->  Walk me through this context in manageable parts step by step, summarizing and analyzing as we go.



《Tree of Thoughts: Deliberate Problem Solving with Large Language Models》

在每一层都可能有不同的选择。在每一步会存在一个value函数，来决策每个节点到当前的路径价值。这个价值函数可以通过投票或者提示词模板生成来实现

![图片](https://mmbiz.qpic.cn/mmbiz_png/4d6kn6AwWwx2Nzm63lVla2siadaCLsvcKIQMI0HQeIJpib3pXwQnC6icic0V0NowNfP02Xj58f587xWztQic5303ichA/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



## CoT相关技巧

https://zhuanlan.zhihu.com/p/669723892

### **如何选取in context learning示例**

相似度retrieval：对准确性要求高，大模型的生成结果严重偏向于演示示例中与输入问题非常相似的问题的标签，类似于复制。（所以有时候zero-shot-CoT效果更好）

Auto CoT：对示例做聚类，每个簇中选一个

Active Prompt：选取模型生成结果不确定的问题，指的是熵比较高、方差高的

Complex CoT：用更复杂的推理作为示例，甚至在解码时多次采样，然后选择复杂推理链中出现次数最多的结果，在逻辑性较强的数据上效果好。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/1BQYN3xneiaaH2O6TIjJQHkkx1pUAno6via7eIkAibAzLByqSdibsXwZ3JL0Gao2icVicHqartymINR5EuZLLsRS9RcA/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)https://arxiv.org/pdf/2310.02954

**「DQ」** 首先通过请求大模型（LLMs）来生成思维链（CoT）。这个过程始于初始的n-shot示例，这些示例可以通过多种检索方法，此外，示例还可以包括人工设计的示例，例如CoT、Tree-of-Thought和Graph-of-Thought等模板。本文主要是采用Complex-CoT方法来获取这些初始示例，因为这种方法能够生成信息量丰富的CoT。

随后，利用这些初始示例和DQ问题再次请求LLMs，从而获得对应的CoT 。最后，结合问题和生成的CoT ，作者使用训练的**「编码器」**来获得测试样本的嵌入表示。

**「Retriever」** 为了获得示例和测试样本的表示，作者训练了一个编码器，同时为了衡量思维链（CoT）与示例之间的相似性，还开发了一个检索器。简单来说，作者利用训练集中的数据来构建训练数据，每个样本由一个问题及其对应的CoT 组成，其中代表训练集中的第个数据点。通过这种方式，编码器能够学习到能够准确反映问题和CoT之间关系的特征表示，从而提高检索器在挑选与输入问题最相关示例时的性能。

**「LoRe」** 在获取了基于语义相似性检索的示例后，使用主成分分析（PCA）进行降维，以去除嵌入中的冗余信息，并使用高斯核函数重新计算示例和目标样本之间的相似性，以此来重排序示例。根据重排序后得到的示例，选择顶部的n个示例，并将它们与问题一起输入LLMs以获得最终的CoT和推理结果

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/1BQYN3xneiaaH2O6TIjJQHkkx1pUAno6vibPPX0dxVyjaaeDuWQtibxTCYjjib7PPA5daCz7c67aSyuDVu6GlvDdaA/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

## LLM评估

### 数据集

**MMLU：**mmlu数据集包含来自各个知识领域的多项选择题。该数据集涵盖了人文学科、社会科学、自然科学以及其他一些对某些人学习至关重要的领域。数据集包括57个任务，其中包括初等数学、美国历史、计算机科学、法律等内容。通过这个数据集可以评估大模型在不同领域的推理能力。

**CMMLU：**CMMLU数据集是一个综合性的中文评估基准，由MBZUAI、上海交通大学、微软亚洲研究院共同推出，在评估语言模型在中文语境下的知识和推理能力方面极具权威性。一句话理解就是中文版本的MMLU。

**C-Eval：**C-Eval是一个全面的中文基础模型评估套件,它包含了13948个多项选择题，涵盖了52个不同的学科和四个难度级别，一句话理解就是中文版本的mmlu。

**GSM-8K：**GSM8K是由人类问题作者创建的8.5K高质量语言多样化小学数学单词问题的数据集，通过这套数据集可以评估大模型的数学推理运算能力。下图是考察大模型8大方面能力，例如写作，人文，推理，角色扮演等，众所周知，数学运算是所有大模型能力最弱的部分。GSM8K数据集就是专门用来评估大模型数学运算能力的。
HumanEval：HumanEval是一个用于评估代码生成能力的数据集，由OpenAI在2021年推出。 这个数据集包含164个手工编写的编程问题，每个问题都包括一个函数签名、文档字符串（docstring）、函数体以及几个单元测试。 这些问题涵盖了语言理解、推理、算法和简单数学等方面。

**MBPP：**MBPP（Mostly Basic Programming Problems）是一个数据集，主要包含了974个短小的Python函数问题，由谷歌在2021年推出，这些问题主要是为初级程序员设计的。 数据集还包含了这些程序的文本描述和用于检查功能正确性的测试用例。一句话理解，和HumanEval一样，也是用于评估大模型代码生成能力的数据集。

**BBH：**一个包含23个具有挑战性的 BIG-Bench 任务的套件，我们称之为 BIG-Bench Hard（BBH）。这些任务是先前语言模型评估未能超越平均人类评分者的任务。

**Multi-HumanEval：**包含多种编程语言的数据集，一句话理解就是HumanEval只包含了python的编程问题，multi-humaneval包含的多种编程语言，例如java，go，javascript等等。

**HumanEval-X：**HumanEval-X 是一个用于评估代码生成模型的多语言能力的基准测试。 它包含了820个高质量的人工制作的数据样本（每个样本都包含测试用例），涵盖了Python、C++、Java、JavaScript和Go这五种编程语言，可用于各种任务，如代码生成和翻译。一句话理解：HumanEval-X数据集和Multi-HumanEval数据集作用相同，只是数据集推出的机构不同而已。

### 评估方法

**选择题类型的数据集**

第一类是选择题类型，例如mmlu，cmmlu，c-eval等都属于这种类型。对于选择题类型的评估数据。原理是：将问题输入给大模型，大模型返回的选择题答案与正确答案进行比较，正确答案的占比作为评估数据的指标值。上面的5-shot指：在输入问题给大模型时，给出5个参考样例。实际每个数据集不同类型的问题中都split出了test/validation/train/dev数据集，其中dev数据集只有5条，这5条通常用于shot数据。

**代码编写类型的数据集**

第二类是代码编写类型的数据集，这里评估的原理是：每一条数据中都带了单元测试，大模型编写的代码与数据集中单元测试进行组合，如果单元测试通过，则认为编写的代码正确。当然，在计算Pass@k值的时候，并不是简单使用通过单元测试的量/总数据集的百分比来计算的。Pass@k有个详细的计算公式，具体如下所示，这里暂不会对公式的计算过程进行详细介绍，在后面源代码解释部分，会详细说明该公式的计算过程。下图是pass@k的计算公式。

![img](https://img-blog.csdnimg.cn/direct/26fc69fbb69949f2b5f4565fceb53478.png)

**数学题目类型**

第三类是从大模型生成的response中提取答案，例如GSM-8K。对于数学类运算问题，在输入给大模型的input text中，通常都会添加step by step的提示语信息。这样，大模型会生成一个计算数学问题的推演过程，在进行评估时，需要从response中提取出最终的答案。然后，和数据集中的正确答案进行对比，从而得到大模型在数学方面的能力分数。
