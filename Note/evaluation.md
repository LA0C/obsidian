
#### 数据集（Llama）
20种数据集，两大设定：zero-shot、few-shot
六大任务：
	常识推断
		BoolQ、PIQA、SIQA、HellaSwag、WinoGrande、ARC easy、ARC challenge、OpenBookQA
		选择填空题，zero-shot
	闭卷问答
		Natural Questions、TriviaQA
	阅读理解
		RACE 
		初高中英语阅读理解
	数学推理
		MATH、GSM8K
		初高中数学题
	代码生成
		HumanEval、MBPP
		代码续写、根据docstring生成、代码润色、单元测试
	大规模多任务语言理解
		MMLLU
		选择题，科学、社会领域知识

#### 指标
##### perplexity
语言模型实质上是计算句子的概率，perplexity是衡量模型生成测试集中句子的概率
PP(w) = P(w1，w2...wn)^(-1/n) = n√(1/p(w1,w2...wn))
相当于是每个字符概率的几何平均，一方面减少句子长度的影响，另一方面必须每个字符概率都较高，结果才会高。