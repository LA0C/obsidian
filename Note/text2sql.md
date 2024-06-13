### 描述

将用户问句转换成表格操作指令，包括

- 筛选
- 排序
- 列管理
- 统计分析
- 绘图

多表融合成大宽表，使整个操作都在一张表上进行，无需多表联查操作。



### 传统方案

意图识别 => 

- 筛选（槽位填充）
- 排序（槽位填充）
- 列管理（槽位填充）
- 统计分析（text2sql）
- 绘图（槽位填充+text2sql）



### prompt方案（大模型function call）

- 先定义function函数名、用途描述、参数名、参数描述
- 大模型的function call根据用户问句分析调用哪个function，返回json包括函数名、参数值



### 拓展

借鉴**data copilot**方案[Data-Copilot: Bridging Billions of Data and Humans with Autonomous Workflow](https://arxiv.org/abs/2306.07209) + https://github.com/zwq2018/Data-Copilot/tree/main

《TableGPT: Towards Unifying Tables, Nature Language and Commands into One GPT》https://arxiv.org/pdf/2307.08674.pdf

chain-of-command：处理复杂、模糊的用户问句，也可以拒绝回答

《TableLLM: Enabling Tabular Data Manipulation by LLMs in Real Office Usage Scenarios》*https://arxiv.org/abs/2403.19318*

- 根据用户问句和数据库进行self-request，生成更多的问句，并定义能解决问句的function
- 调用function解决问题，将原始数据转换成结构化数据，如图形、表格、文本

和金融agent方案：https://github.com/Tongyi-EconML/FinQwen/tree/main





