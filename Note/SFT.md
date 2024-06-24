https://mp.weixin.qq.com/s/6hs39qm4_WhdGhuuQIxiUQ
https://zhuanlan.zhihu.com/p/695378820
### 1.常见SFT开发流程
#### 根据业务场景编写提示词：
- 定义越详细越好（例如分类标签的定义）
- 不要让模型理解歧义（符号信息要一致，段落区分不要只用简单的\n）
- 遵循system message，input，instruction三段式
- 通过大模型输出解析内容，用于调整prompt
#### 尝试各种开源闭源的模型
llama2、Qwen、baichuan、chatGLM，了解不同模型擅长点
#### 准备数据集
通常每个子任务的数据量不超过1k条，要包含任务的边界样本和困难样本，确保数据多样性和标签的平衡。主要三个方面：生成数据质量评估（准确、简洁）；多样性筛选（指令类型、难易程度、复杂度）；必要性（相似问句答案不同、噪声数据、模型训练前回答效果差的数据）
#### 上线迭代
持续迭代更新

### 2.训练数据需要注重什么
#### 确保回答格式和风格统一
先复述一遍问题、再回答、在总结
#### 数据均衡
难易样本均衡，数据多样性和标签的平衡
可借鉴llama2和Qwen技术报告中关于sft数据的经验https://mp.weixin.qq.com/s/3MQdDabjX0rIX0OozxYk3g、https://mp.weixin.qq.com/s/rHJkJw9TFGaAR8bWDM5wmg



### 3.模型size
尽量选大的，容错率更高
### 4.多任务训练时不同任务间的影响
实践显示任务间的影响是正常现象
应对方法：不同任务独立训练
任务取舍：全部任务训完之后，对某些关键任务额外多训一个epoch

### 5.SFT能否学到知识
常识和世界知识难以通过sft灌输给模型，比如预训练只更新到NBA2022年总冠军是勇士，sft训了QA对：nba2023总冠军是谁，掘金。再让模型总结近几年NBA总冠军，还是只能总结到2022年，或者问它掘金是否获得过NBA总冠军，答案还是否。
sft应该关注激发模型预训练中学到的知识，学习业务相关的特定规则、输出格式稳定。

### 6.如何科学挑选数据集
大部分业务场景，几百条数据足以
[LLM2LLM：迭代数据增强策略提升大模型微调效果](https://mp.weixin.qq.com/s?__biz=Mzg5MTU1NTE1OQ==&mid=2247488950&idx=1&sn=b4ca115d1c7ac41557668a6c72e7eb59&scene=21#wechat_redirect)、[自我蒸馏方法-减轻大模型微调过程中的灾难性遗忘](https://mp.weixin.qq.com/s?__biz=Mzg5MTU1NTE1OQ==&mid=2247488268&idx=1&sn=b4b0d165fab9b546ef543b597dac6b14&scene=21#wechat_redirect)、[DEITA-大模型指令微调的数据高效筛选方法](https://mp.weixin.qq.com/s?__biz=Mzg5MTU1NTE1OQ==&mid=2247487947&idx=1&sn=0b23e924df987565c4808f74dbf79ddd&scene=21#wechat_redirect)、[大模型微调技巧 | 高质量指令数据筛选方法-MoDS](https://mp.weixin.qq.com/s?__biz=Mzg5MTU1NTE1OQ==&mid=2247487747&idx=1&sn=f11dafe28e8ceb0faa38c9d55a262633&scene=21#wechat_redirect)、[如何从数据集中自动识别高质量的指令数据](https://mp.weixin.qq.com/s?__biz=Mzg5MTU1NTE1OQ==&mid=2247486951&idx=1&sn=e4bd058dbe9c191ecfeec3c75e8bff06&scene=21#wechat_redirect)

### 7.如何决解幻觉问题
过度联想问题，还是要用SFT、强化学习来解决

### 8.BERT和LLM训练的区别
LLM需要的数据更少，整体训练集：bert要近万条，LLM只要1k左右；bad case改进数据：bert要几十上百条，LLM只要10条

### 9.如何选择full-tuning、P-tuning、Lora
SFT scaling law：训练集几千条用P-tuning；上万条用Lora；百万条用Full-tuning。
大多数场景下还是用lora，更稳定、保持原有模型的泛化性

### 10.还有哪些方面值得研究
消除幻觉、任务配比（通用和领域5:1）、更高效的微调方法（lora+，qlora，Plora）




一、通用Finetune方法
数据去噪：通过规则、共现性
数据筛选：
	质量：人工标注之后训了个打分模型
	多样性：k-means聚类，类别内数量多的通过语义相似度去重
	必要性：根据模型训练效果，增加效果差的，不断迭代

二、LLM自动生成指令数据
Self-Instruct: Aligning Language Models with Self-Generated Instructions（[paper](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2212.10560)、[github](https://link.zhihu.com/?target=https%3A//github.com/yizhongw/self-instruct)）
准备任务指令库，每个任务有一个示例指令，生成新指令，识别是否是分类任务，分类采用输出优先prompt，非分类采用输入优先prompt。最后过滤掉低质量、重复的指令。

Evol-instruct：https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2304.12244.pdf
从广度和深度上扩展指令
*结合AI报表*：广度上替换不同类型的表格和表格字段；深度上通过增加约束、推理步骤、输入复杂度。

三、LLM评估
保留测试集，便于模型迭代之后的快速效果测试。

