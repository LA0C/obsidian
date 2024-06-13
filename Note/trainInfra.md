## 1.分布式并行策略

### 常用策略

- 数据并行：每张卡都保存一个模型，每次将batch数据分成mini batch分别计算，在通过allreduce汇总
- 模型并行（张量并行）：将有些很大的tensor切成多块
- 流水线并行：将网络按层切分

对数据并行进行优化，减少冗余参数，通过分片，每张卡只存1/N的模型状态量，整体就只维护一份模型状态量

这样会产生额外通信，但可以通过allreduce优化



Zero是deepspeed提出的分布式训练框架

在DeepSpeed下，ZeRO训练支持了完整的ZeRO Stages1, 2和3，以及支持将优化器状态、梯度和模型参数从GPU显存下沉到CPU内存或者硬盘上，实现不同程度的显存节省，以便训练更大的模型。

![image-20240117103340635](https://s2.loli.net/2024/01/17/ZFuLfk9ODhd38Ri.png)

一般来说，显存上有三种数据：模型参数、梯度和优化器参数。

例如，一个7.5B的模型，如果用fp16训练，则一个参数需要占用2个字节，7.5B需要占用7.5*2=15GB字节。

梯度会用fp32计算，但用fp16保存，则也需要15GB字节。

如果优化器选用Adam，需要用32位来计算，则每个参数需要4个字节，另加4字节的momentum和4字节的variance，一共12字节，则7.5B需要占用12 * 7.5B = 90GB字节。

这样一共就是120GB。

**Zero1**就是将优化器参数切割，放到N=64张卡上去，这样显存占用就降到了31.4GB。

**Zero2**再将梯度进行切割，这样又降到了16.6GB。另外可以添加**optimizer offload**，将优化器参数和梯度下沉到CPU上。

**Zero3**再将模型参数也切割了，最后一张卡上仅需要1.9GB。另外可以添加**parameter offload**，将模型参数也下沉到CPU上。但是Zero3的通信量会增加（**正常是2![image-20240601174011133](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240601174011133.png)，参数需要从各个卡上汇集一次，多了**![image-20240601174103744](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240601174103744.png)），在4卡3090的实验上发现，Zero3的训练速度远低于Zero2，推测是3090的带宽较低。

