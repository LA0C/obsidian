
# Model
***BGE***：https://arxiv.org/pdf/2309.07597, https://zhuanlan.zhihu.com/p/670277586
![image-20240520163304605](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240520163304605.png)

enhanced decoding：
除了第一行，每个token可以看见第一个元素和随机采样的词

训练策略：
- 通用文本上预训练（数据量TB；RetroMAE，decoder掩码率更高70%，更难）
- 通用文本上finetune（100M unlabeled data，对比学习）
- 特定任务上finetune（800k labeled data）



### Visualized BGE
- 文本检索多模态（query：text;  candidate: image-text text image）e.g. [WebQA](https://github.com/WebQnA/WebQA)
- 多模态检索图像（query：image-text;  candidate: image）e.g. [CIRR](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/visual), [FashionIQ](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/visual)
- 多模态检索文本（query：image-text;  candidate: text）e.g. [ReMuQ](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/visual)

##### EVA-CLIP架构：
image-encoder编码图像a
text-encoder编码文本b
同一个（图像-文本）pair的a和b更近
eva-clip与clip的差别：训练层面的优化（eva模型初始化、优化器）


### BGE-M3
多语言、多粒度（输入长度达8192tokens）、多功能（dense、sparse、multi）
数据：1.2亿多语言的文本对，包含有无标签和生成数据（根据文档生成可能的问句）。
	1.没有标注信息的弱监督数据：来自于从网上挖掘得到的各种有语义关联的数据，并过滤掉其中低质量的内容。
	2.来自有标注信息的监督数据：包括若干个中文跟英文的开源数据集，例如MS MARCO，NLI，DuReader等。
	3.合成得到的监督数据：利用GPT3.5为来自Wiki跟MC4的长文本生成对应的问题，用于缓解模型在长文档检索任务的不足，同时引入额外的多语言数据

检索方式：dense计算内积、
                  sparse计算重复token加权得分，权重由token的last-hidden-states经过Relu激活得到、
                  multi-vec计算query每个token和文档每个token的向量相似度，取最大值为得分，所有位置得分求平均，就得到相似度得分了。其中每个token的向量通过对last-hidden-states做全链接和norm
训练方式：
	loss由两部分组成：infoNCE
		自知识蒸馏：三种方式的相似度加权和作为teacher得分，让三种方式去学习，再对三个交叉熵loss求和平均
	多阶段训练：
		RetroMAE做预训练，获得一个底座
		二阶段训练只考虑dense检索的infoNCE
		三阶段在有监督数据2.3上训练
	hard负样本：
		ANCE：语料库的最邻近(ANN)索引构造负样本，表示向量的encoder每步都会更新。


# application

### 小布助手召回排序链路：
![图片](https://mmbiz.qpic.cn/mmbiz_jpg/jT5Sp7MICPyh6oQuI6tGoLsUGhbkfQ8bz2jstf4QolLRkUAZT9YBn54f7ESUklpEXAWKZC8stSxY83zVUtn2fw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

## 字面检索的优缺点

字面检索作为最早期搜索的主流方法，即使在今天，也没有被完全淘汰，自然是有他的很多优势的，先列举一下：
- 字面的可解释性高，对应的也是可控性强。
- 准确率相对高，召回的内容可靠程度高。
- 字面检索对很多基础工作有要求，算法上例如同义词、紧密度、改写之类的，工程上需要完善的倒排索引，但是这些问题的解决方案其实都相对成熟。
- 快速构建，对比向量检索可以不需要模型支撑（向量检索刚需表征模型），这也让性能需求降低，上线压力小。
- 基础的检索，不需要进行模型训练，不需要训练数据。

但是，当然也是有缺点的，我也列举一下：

- 局限在字面，因此泛化能力刚需各种支持工作，例如同义词、紧密度、改写等，所以天花板较低，而且后续的维护和提升基本都局限在这里。
- 基础工作的展开，类似意图识别、NER之类任务的加入，是有利于就准确率提升的，但是就是会有依赖。
- 字面依赖高后，不分内容召不回会导致需要构造更多的文档来保证答复率或召回率，因此文档数量会提升，存储压力加大（当然这个量可能早期不会很高）。
- 泛化能力不足，召回率不足。

这么看，评价起来其实就是字面检索其实有很高的下限，在项目初期的时候其实挺推荐用字面检索的方案来处理大量的问题，而且可控性高。

## 向量检索的优缺点

向量检索是现在大家都聊的比较多的问题了，口碑没什么问题的。那么展开聊一下他的优劣势吧。首先是优点：

- 精准度尚可，泛化能力高，能匹配到很多同义词、新说法之类的。
- 相比字面检索，对基础工作的要求不会很高，同义词、紧密度、改写等要求不高（当然在模型训练是还是可用来做数据增强的）。

虽然有点比较少，但是这两个收益就已经足够大，前者其实能让有限的数据被更多召回，而后者看上去则降低技术工作的成本，这也是很多人其实更倾向于选择这个类型的方法作为首选，但其实会有些缺点：

- 强依赖表征模型的效果，下限很低。
- 开放域固然可以使用训练好的开放域模型，但是对于自己的特定领域，或者有些业务需求，是需要依赖数据的。
- 调优成本高，模型训练、模型切换、重构索引等，可控性低。
- 因为要模型，性能这个问题自然就要被关注起来。
- 有些特定问题向量检索不合适或者不优先，例如“筛选年龄18岁以上的人员”之类的检索要求。

可以看到，其实这是个不太泛用且不太可控的方案，但其实因为很多客观原因可能并非是真正的万能解。一分为二吧，其实更像是一个在某些场景下锦上添花的方案，能在一定场景下快速提分。

## 两者的选型策略

我对这俩整体的选型策略，其实和之前我一些别的问题里的思路是类似的：

- 初建项目，首先考虑字面检索。
- 逐步完善字面检索的各种基础工作，确保准确（宁可不出，不要出错）
- 随着泛化能力的逐渐暴露，开始考虑上向量检索。两套方案并行形成多路召回。
- 优化两者协同性，开始考虑重点优化精排等部分