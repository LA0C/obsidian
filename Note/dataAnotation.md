## 大模型用于数据工程

### 1.用于数据标注

《Large Language Models for Data Annotation: A Survey》(https://arxiv.org/pdf/2402.13446)

论文集：https://github.com/Zhen-Tan-dmml/LLM4Annotation.git

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/fUBU1yiaEmJiaJ9gldgIrYtOEADDXOqg9yvyo5zHzphvbJ2lbvZC9hv1BFtleWI8XiclcvibUzBiaQibxle0zwBgJSiaA/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

### 2.大模型用于知识蒸馏

大型语言模型(LLM)的知识蒸馏(Knowlege Distill)技术，通常将GPT-4等专有大模型的复杂能力转移到开源模型(如LLaMA和Mistral)中，这是现在主流self-instruct中常用的一条路

《A Survey on Knowledge Distillation of Large Language Models》(https://arxiv.org/abs/2402.13116)

论文集：https://github.com/Tebmer/Awesome-Knowledge-Distillation-of-LLMs

**根据蒸馏对象区分：**

标注Labeling:教师从输入中产生输出；

拓展Expansion:教师通过情境学习生成与给定演示相似的样本；

数据整理Data Curation:教师根据元信息，如主题或实体，对数据进行综合；

特征提取Feature:将数据输入教师，提取其内部知识，如逻辑、特征等；

反馈 Feedback:教师对学生的代际进行反馈，如偏好、纠正、扩展具有挑战性的样本等；

自我认识Self-Knowledge:学生首先生成输出，然后筛选高质量或由学生自己评估

**根据蒸馏方案区分**：

IF:指令遵循，MD:多轮对话，TP:思维模式，RAG:检索增强生成，NLU:自然语言理解；

NLG:自然语言生成，IR:信息检索，SFT:监督微调，D&S:发散与相似，RL:强化学习，RO:排名优化

