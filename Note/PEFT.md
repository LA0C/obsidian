### P-Tuning, Prefix-Tuning, Prompt-Tuning

在原始模块中添加adapt、promptEncoder模块，训练时参与，推理时就固定住。

### LoRA

原始模型的权重通常是满秩，但语义信息往往存在于低秩。

在训练时，采用低秩分解来模拟参数的更新量

**调参**：秩r，在Q V中应用

**实现：**训练时，分别计算AB和原始线性层的结果，并相加

​           推理时，merged=True，将AB矩阵相乘的结果合并到原始权重上

### AdaLoRA

根据模块重要性自适应的调整秩（奇异值分解，根据重要性不断减小秩的大小）

### QLoRA

模型原权重量化成4bit，计算时反量化成16，跟lora的精度保持一致。

增加adapter，相比lora只在Q V中添加，qlora在所有全连接层都插入了adapter

