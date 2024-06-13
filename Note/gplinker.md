## 金融事件抽取：

| 事件类型 | 事件角色 | 事件角色   | 事件角色 | 事件角色   |
| -------- | -------- | ---------- | -------- | ---------- |
| 公司上市 | 公司名称 | 证券代码   | 时间     | 价格       |
| 股东减持 | 股票简称 | 减持方     | 时间     | 减持比例   |
| 股东增持 |          |            |          |            |
| 企业收购 | 收购方   | 被收购方   | 时间     | 收购金额   |
| 企业融资 | 投资方   | 被投资方   | 时间     | 融资金额   |
| 企业破产 | 公司名称 | 债务规模   | 时间     |            |
| 企业亏损 | 公司名称 | 亏损规模   | 时间     |            |
| 高管变动 | 高管姓名 | 变动前公司 | 时间     | 变动后公司 |



## GPLinker

**GlobalPointer**：通常pointer network使用两个模块分别识别实体的首和尾，GP将首尾视为一个整体。

构造一个上三角矩阵，每个点都是一个序列片段，一共有n(n+1)/2个，其中可能有k个实体（假设只有一类实体），此时就转换为n(n+1)/2选k，m类实体就是m个n(n+1)/2选k

![image-20240527195210810](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240527195210810.png)

q和k表示第α类实体的向量序列，内积作为片段是否是实体的打分函数，简化版的MHA，有多少类实体就有多少个head，只是少了V的运算。

**相对位置信息**：RoPE

**多标签分类交叉熵**：多标签分类传统做法是拆成单标签sigmoid激活的二分类，存在类别不均衡（大多数都是负样本），而单标签分类可以用softmax，不存在类别不均衡。于是将softmax推广到多标签分类。目标类别得分与非目标类别得分之间两两比较，logsumexp平衡了每一项的权重。

**Efficient GlobalPointer**：参数太多，有α个Wq和Wk，可以分解为抽取和分类两部，抽取共享q、k参数，分类采用特征拼接+Dense层来完成。

![image-20240527204051959](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240527204051959.png)



## Contrastive Learn

主要思想是利用不同的数据增强方法生成正对，以两个独立的样本作为负对，然后采用InfoNCE损失拉进正对的嵌入，疏远负对的嵌入
现有构造方法的问题：
- 长度敏感：simcse中通过dropout构造的正对都是长度一样的，会让模型认为长度一样的才更相似
- 相似度边界：“上证指数今天跌幅扩大至2%”和“深证指数今天跌幅扩大至2%”
- 数据增强：词语删除、替换和重复等操作，缺乏导向性、无法控制样本的正负极性，且多样性不足。
方案：根据业务场景定义边界，构造提示模板；借助LLM生成正负样本；训练得到文本语义模型。
边界：不同行业的公司，发生相同事件时，属于不相似事件
模板：将下面句子中的公司实体换成同一行业的其他公司，并换一种表述方法（正样本）
​           将下面句子中的公司实体换成不同行业的其他公司，并换一种表述方法（负样本）
训练：增强数据混合通用数据，RoBERTa
​           InfoNCE损失（温度系数的作用就是控制**模型对负样本的区分度**）：
​          ![image-20240528200842595](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240528200842595.png)
​			双向边际损失，控制相似度差异在![image-20240528201739671](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240528201739671.png)区间内：
![image-20240528201250075](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240528201250075.png)
​			deta表示原始样本与正样本以及原始样本和负样本之间的差异，
![image-20240528201604423](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240528201604423.png)

