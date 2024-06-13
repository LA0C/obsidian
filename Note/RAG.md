## **paper**
[self-rag]: https://zhuanlan.zhihu.com/p/666074661
自我反思检索增强生成，提升语言模型生成质量与准确性

[Gorilla]: https://zhuanlan.zhihu.com/p/640697382
与大规模API相连的大型语言模型
否加入api检索器来推理api调用指令，发现检索器不一定能提高效果，检索结果会误导llm

[ToolLLM]: https://arxiv.org/abs/2307.16789
[multi-agent蒸馏到single]: https://arxiv.org/pdf/2401.07324
[多Agent变成单一模型]: https://arxiv.org/pdf/2404.04619.pdf


## langchain
https://zhuanlan.zhihu.com/p/651151321
models、prompt、index、chain、agent、memory

##### langchain-RAG
query translation：问句拆解、HyDE
Routing：通过语义选择数据库或prompt
Query construction：将文本转换成检索需要的格式，text2sql，text2emb
Indexing：chunk切分方式优化、对doc做summary、多种索引结构
Retrieval：检索和压缩
Generation：自适应adaptive、循环iterative


## 报告
#### 1.背景
llm的局限性
- 幻觉
- 信息过时
- 低效的知识参数化
- 缺乏专业领域知识
- 推理能力较差

应用层面的需求
- 深度专业精确答案
- 数据更新
- 可溯源和可解释性
- 结果可控，数据安全

#### 2.RAG介绍
rag就是检索相关信息来辅助大模型回答
![图片](https://mmbiz.qpic.cn/mmbiz_png/vtIvcrPJjh6LwXNicqj4G2tibibibAYyiaOWcgk80NW36AJ86vicBnbdkHD2DL6RnMsIwbSgWDSrTzOI71MnjBk4c1NQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

#### 3.大模型优化方法
Prompt Engineering
Retrieval-Augmented generation（动态知识更新、领域知识库、可解释性、可溯源、减少模型幻觉）
Fine-tuning

#### 4.应用场景
QA、Fact checking、Dialog、Summary、Machine Translation、Code Generation、Natural Language Inference、Sentiment analysis、Commonsense reasoning
![image-20240513095215850](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240513095215850.png)

#### 5.rag框架
- Naive RAG
- Advanced RAG
- Modular RAG
##### <u>Naive RAG</u>
![image-20240513101429141](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240513101429141.png)

**step1 Indexing（离线）**
文档切块、embedding、存储vector database
**step2 Retrieval**
user query和切块计算embedding相似度，最相关的K个文档
**step3 Generation**
query和检索结果合并成prompt，输入LLM



##### <u>Advanced RAG</u>
![image-20240513101743617](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240513101743617.png)

**Pre-Retrieval**
细粒度切分：粗粒度的文本信息转换向量后，很多信息会被稀释。可以考虑大纲、章节标题等细粒度。
多样化索引：列表索引（原始）、树索引（单个文档）、关键词索引（将query快速路由到不同的数据源）、摘要索引（提取关键信息，提高性能）
对齐优化：指检索的内容和用户的提问尽可能对齐，这样计算的相似度会比较高，容易被检索到。

**Post-Retrieval**
re-rank ：根据文档多样性重排，精排embedding模型（更准，但是慢）
prompt压缩：生成之前先压缩，减少噪声，突出关键信息。计算互信息和困惑度（Anderson et al. *2022Proceedings of the 15th Biennial Conference of the Association for Machine Translation in the Americas.*），训一个小模型（Xu et al.Improving retrieval-augmented lms with compression and selective augmentation）

##### <u>Modular RAG</u>
考虑更多的模块
![图片](https://mmbiz.qpic.cn/mmbiz_png/vtIvcrPJjh6LwXNicqj4G2tibibibAYyiaOWcibTUqYjVjgSfcI0C39e9cRRZjKav8g6P0N98986rleNt6fmibEfHhBwQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![image-20240517142437362](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240517142437362.png)

- memory module：记忆之前的问题和输出，可用于指导检索和生成
- fusion module：user query变成多视角多样化的multi-query
- routing：查询路由决定对用户查询的后续操作，选项包括摘要、搜索特定数据库或将不同的路径合并为单个响应
- predict：训一个或多个模型，来生成特定领域的内容，作为检索结果。（ner、情感分类）
- task adapter：构建zero-shot和few-shot prompt库，针对不同任务搜索最适合prompt来帮助llm执行task
- **模块组合更灵活，并不是模块越多越好，要在检索效率与上下文信息的深度之间做平衡**

#### 6.Retrieval

检索模块是rag重点，主要有3个关键任务：
- 如何获得准确的语义表示
- 如何匹配query和文档之间的语义
- 如何让大模型和检索结果更好的协同

##### 如何获得准确的语义表示
分块和微调
分块：根据索引内容的特点、embedding模型的效果、query的复杂程度来决定。
​          滑动窗口和small2big的方案
​		为每个chunk生成一组合成问题，直接和query进行匹配（https://mp.weixin.qq.com/s/LDUDOG9u7lop_bGlnlP4Mg）

微调：领域知识训练（queries、语料库、相关文档），下游任务微调（标注硬标签、软奖励）

***BGE***：https://arxiv.org/pdf/2309.07597
https://zhuanlan.zhihu.com/p/676410726
![image-20240520163304605](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240520163304605.png)

训练策略：
- 通用文本上预训练（RetroMAE，decoder掩码率更高70%，更难；）
- 通用文本上finetune（100M unlabeled data，对比学习）
- 特定任务上finetune（800k labeled data）

##### 如何匹配query和文档之间的语义
我们知道，query和doc本身的语义并不是相似，只是相关，所以需要通过改写、归一化让两者更好的匹配起来
query侧：query2doc（Wang et al.*Query2doc: Query expansion with large language models.*）
​                 HyDE（Gao et al.*Precise zero-shot dense retrieval without relevance labels*.）
![image-20240520105023665](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240520105023665.png)

![image-20240520102529692](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240520102529692.png)

![image-20240605194852217](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240605194852217.png)
doc侧：问题挖掘，特征抽取

##### 如何让大模型和检索结果更好的协同
检索结果正确，但大模型生成的仍然不好，这是因为检索到的文档可能不符合大模型的要求，需要做对齐训练。
**retrieval微调**主要思路：利用来自llm的反馈信号（对不同检索结果）来改进检索模型
​                   Cheng et al.*Uprise: Universal prompt retrieval for improving zero-shot evaluation.*
​                   Shi et al*.Replug: Retrieval augmented black-box language models.*
**adapter**：根据不同任务来整理相应的关键信息Xu et al. *Recomp: Improving retrieval-augmented lms with compression and selective augmentation.* 



#### **7.Generation**

- 检索后处理
- 生成器微调

##### 检索后处理
**信息压缩**：llm的上下文长度有限，且较长输入会降低效果，所以要抽取关键信息，过滤无关信息。
Xu et al.*Recomp: Improving retrieval-augmented lms with compression and selective augmentation.*
生成之前先压缩，减少噪声，突出关键信息。计算互信息和困惑度（Anderson et al. *2022Proceedings of the 15th Biennial Conference of the Association for Machine Translation in the Americas.*），训一个小模型（Xu et al.Improving retrieval-augmented lms with compression and selective augmentation）

**re-rank** ：根据文档多样性重排，精排embedding模型（更准，但是慢）
与召回模块的独立embedding再算相似度的方案不同，rerank主要采用交互式。
![img](https://pic1.zhimg.com/80/v2-9e1b18c1116e7385843f2db1ed7f6810_720w.webp)



##### 生成器微调
构造input，output pair来进行有监督的学习，input可以是正确的 也可以是不正确的，从而减少“曝光偏差”，让模型接触到更丰富的反馈，提升鲁棒性。
![image-20240520105208786](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240520105208786.png)

#### 8.RAG调优

- 调优的阶段
- 数据来源
- 调优思路

##### 调优的阶段
预训练阶段：加入领域知识，训练任务采用rag模式
微调阶段：以上说的对检索器和生成器的微调，对数据质量的要求比预训练更高
推理阶段：prompt优化，cot tot，无需更新模型参数

##### 数据来源
非结构化：文本数据
结构化：知识图谱
LLM生成：大模型自己创造内容来辅助自己生成，生成一些特殊符号，用来判断内容是否可靠、是否需要检索、选择何种回复策略（self-rag）

##### 调优思路
迭代检索和自适应检索（agent思路，通过多次调用llm，来判断什么时候查询、查什么、怎么查）
<u>*adaptive*</u>
step1：判断是否需要检索，是的话生成一个“retrieval” token 按需检索
step2：并发检索多个段落，分别生成结果
step3：生成“critique” token（retrieve|IsRelevant|issupported|isUse），从事实性和整体质量方面选择最好的生成结果（llama2 7B 13B 150kdata）
![image-20240520105437131](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240520105437131.png)



<u>*iterative*</u>
GAR：用query+generation(t-1)去检索
RAG：refine：上一次生成结果+query+当前检索 - 上一次检索
​           refresh：query + 当前检索
![image-20240520213348052](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240520213348052.png)

缺失信息引导的检索-抽取-解决框架https://zhuanlan.zhihu.com/p/694229570

#### 9.RAG评估
##### 评估方法
- 独立评估：检索（命中率、平均排名倒数、NDCG）；生成（rouge、相关性、非有害性、真实度）
- 端到端评估：生成效果，人工标注
##### 关键指标
答案准确性：与上下文一致，符合事实
答案相关性：所答对应所问
抗噪声能力、拒绝无效回答能力、信息综合能力等等

##### 评估框架
RAGAS和ARES
Es et al. *Ragas: Automated evaluation of retrieval augmented generation.*
Saad-Falcon et al. *Ares: An automated evaluation framework for retrieval-augmented generation systems.*

#### 10.未来展望
##### 模态、内容拓展
图像、文本、代码、音视频
##### RAG生态
对相关技术栈和工具的要求变高，参考Langchain和LlamaIndex
**多模态方向：**
首先基于视觉信息生成中间内容作为理由，然后使用生成的理由生成结果；
构建多模态知识索引。