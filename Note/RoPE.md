## 旋转位置编码

### 1.位置编码介绍

https://mp.weixin.qq.com/s/XDogAZOaWu09k_qLs9tMZA

**绝对位置编码**：就是用 位置索引 直接进行编码, 一般都是直接构建 位置嵌入向量, 和 词嵌入向量 直接相加。Transformer 中的 正弦波位置编码, BERT 和 GPT 中的 可训练式位置编码 都属于这一种

**相对位置编码：**

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/tA8ibKWwC0Gyp0n6c180y1olpHnSJBkeOjbdtLic5ibTSvxmtUY043SxNbpLVhSTXfRRtxMUnWDtFrmITmx16POAw/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



#### Sinusoidal位置编码

d表示词向量的维度，k表示位置索引，下面公式分别表示位置K的位置向量的第2i和第2i+1个分量：

![image-20240601135245100](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240601135245100.png)

**周期性**和**远程衰减性**

### 2.推导

将f(q,m)和f(k,n)表示成关于q、k、m-n的函数

首先找到f()这种表示方法：

![image-20240601141002352](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240601141002352.png)

![image-20240601143122653](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240601143122653.png)

等于对向量做弧度为θ的逆时针旋转。

**代码：在原来attention中，先定义好旋转矩阵(dim, seq_len*2)，在计算attention前先对向量进行旋转编码**



借助sinusoidal，通过![image-20240601141513340](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240601141513340.png)引入远程衰减性

10000称作base，与长度外推有关，NTK本质上都是通过放大base来改变旋转角度，进而影响模型的位置编码信息，最终达到长度外推。
base变大，周期性会变长

https://zhuanlan.zhihu.com/p/647109286

https://zhuanlan.zhihu.com/p/644585013

### 3.直接外推与线性内插

**直接外推**可以推理，但效果不好，与预训练的位置信息不一致

**线性内插（position interpolation）**

将位置索引m缩放到训练的范围内，也相当于是扩大了base

![image-20240601151402472](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240601151402472.png)

在长样本训练1000步就有好的效果



**NTK-aware scaled RoPE**

![image-20240601154234114](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240601154234114.png)

PI是将旋转速度变慢，但应该有所区分，NTK将base扩大α倍（进制转换的思想），实现高频（短距离）外推、低频（长距离）内插



**NTK-by-parts**

根据波长来进行区分，旋转一周所需的长度，波长小外推，波长大内插。

**Dynamically NTK**

![image-20240601161204557](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240601161204557.png)

**YaRN**

在NTK基础上，再通过温度对attention score进行调整



**其他**

window attention、streamingLLM、LongLoRA等