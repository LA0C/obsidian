![图片](https://mmbiz.qpic.cn/mmbiz_png/GmyBmIxnRkMAFJTU4OcjZgBQI9a5Dkv1CkNwPxSZdfeUxZ7mmlNYtkE5IiaBLmQvheJZwUgu8QYZywUe3WzDY5A/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

## RLHF流程

1.标注数据训练一个sft model

2.pair打分数据训练reward model（计算即时收益）

3.ppo方式训练policymodel（目标LM）, reference model（加上约束，防止寻歪了）和critic model（来自RM，预估总收益）
   3.1 初始化参数θ和θ'（off-policy策略，交互和学习是不同步的，θ'负责交互产生数据来更新θ，这样更节省资源和时间）
   3.2 用θ'与环境交互产生状态行为数据（st,at），计算advantage Aθ'
   3.3 使用ppo目标函数来拟合θ（奖励部分：*importance weight和优势* 和 KL部分）

![image-20240529133008927](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240529133008927.png)

### loss计算

#### 1.actor loss

##### （1）直观设计

评估actor输出是否符合人类偏好

![image-20240428143412026](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240428143412026.png)

Vt表示总收益，P(At|St)表示根据当前上文S产生token A的概率

求和表示只考虑response部分的token，去掉之后简单表示为

![image-20240428143822847](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240428143822847.png)

##### （2）引入优势 Advantage

优势表示实际收益超出预测收益的部分，为何要用优势？引入Rt？

critic对At的收益预测为Vt，实际收益为![image-20240428144057458](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240428144057458.png)

优势为：![image-20240428144139028](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240428144139028.png)（即时奖励+衰减的未来预估收益 - 当前的预估收益）

此时loss为：

![image-20240428144159933](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240428144159933.png)

##### （3）重新设计Rt

Rt表示At带来的即时收益，在deepspeed-chat中改造为：

![image-20240428144803234](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240428144803234.png)

kl_ctl是常量，缩放因子，默认为0.1

log部分是kl散度

公式可以理解为：

***当t！=T时，只关心actor在ref的约束下生成At***

***当t=T时，不仅关心约束，也关心最后时刻的即时收益Rt（也可替换成所有token收益的平均值）***

##### （4）重新设计优势

引入了对未来优势的考量，将

![image-20240428144139028](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240428144139028.png)

改写为

![image-20240428145511091](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240428145511091.png)

注意这里的At+1不是token，而是优势Advt+1，计算方法采用动态规划，因为最后时刻T的未来收益Vt+1和未来优势At+1都是0，此时![image-20240428145927057](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240428145927057.png)

##### （5）PPO-epoch：引入新的约束

当前整个ppo的训练过程如下

![图片](https://mmbiz.qpic.cn/mmbiz_png/GmyBmIxnRkMAFJTU4OcjZgBQI9a5Dkv1eWKicVzfTbfSqQuOLMz6aWdt8RcFzVmb1oF0CqlaqHicoXFOlAImFLkg/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

***第一步：输入一个batch的prompts***

***第二部：Actor生成responses***

***第三部：prompt+response喂给Critic/Reward/Reference，生成用于计算loss的数据，称为经验（experiences）***

***第四部：计算loss，更新Actor和Critic***

第四步中，1batch的经验exp，用来计算了ppo-epochs次loss，也就是更新了ppo-epochs次的Actor和Critic

等价于Actor模拟和环境交互了ppo-epochs次：

***如果1batch的exp只更新一次，那么更新完后，Actor就吃新的batch，和环境交互产生新的exp***

***如果1batch的exp被使用ppo-epochs次，在ppo-epochs次中Actor是不吃任何新数据，也不和环境交互的，只是模拟交互，产生一些新的数据。***

如何模拟？观察一下之前的数据，依葫芦画瓢。最开始吃batch经验的actor叫Actor(old)，ppo-epochs中的叫Actor(new)，只要保证new能模仿old就行，保持两个分布相近，自然就是**KL散度**

actor_loss可以改进成：

![image-20240428151917763](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240428151917763.png)

再稍作一些改动将log去掉（这个其实不是“稍作改动去掉log”的事，是涉及到PPO中重要性采样的相关内容可以参考https://www.cnblogs.com/xingzheai/p/15931681.html）

![image-20240428152002454](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240428152002454.png)

为了避免分式的值超过可接受的范围，需要clip一下

![image-20240428154936910](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240428154936910.png)

actor_loss小结：

- ![image-20240428155140000](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240428155140000.png)

- Rt能够衡量Actor是否遵循Ref的约束

- Advt能够考虑当前时刻和未来的优势advantage
- 1batch的数据做ppo-epochs次模型更新，并采用clip控制范围，超出范围则不更新



#### 2.Critic loss

实际收益 - 预估收益

![image-20240428195511462](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240428195511462.png)

原始的实际收益为![image-20240428195546491](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240428195546491.png)

引入优势：![image-20240428195700712](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240428195700712.png)

这里不懂，优势Adv不就是







## DPO

instructGPT中混合了pretrain的梯度，称作“PPO-ptx”![image-20240528214544110](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240528214544110.png)

rlhf代码解析:https://zhuanlan.zhihu.com/p/650505023

详解：wx里面找



#### RLHF的两个阶段

step1：reward model

![image-20240528210511382](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240528210511382.png)

![image-20240528210610645](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240528210610645.png)是reward model，x是prompt，ywin和ylose是好的和坏的回答。让好回答的得分高于坏回答。

​		step2：ppo

![image-20240528211339276](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240528211339276.png)

![image-20240528211607140](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240528211607140.png)是要训的模型，![image-20240528211633974](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240528211633974.png)是参考，在最大化奖励的同时，不要偏离原始模型太多。

#### dpo推导

上式通过构造

![image-20240528213212469](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240528213212469.png)

变换成

![image-20240528213238424](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240528213238424.png)

KL散度在![image-20240528211607140](file://C:/Users/viruser.v-desktop/AppData/Roaming/Typora/typora-user-images/image-20240528211607140.png?lastModify=1716903251)和![image-20240528213430297](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240528213430297.png)相等时取最小值，得到结论：最优的分布就是![image-20240528213430297](file://C:/Users/viruser.v-desktop/AppData/Roaming/Typora/typora-user-images/image-20240528213430297.png?lastModify=1716903269)

![image-20240528213629722](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240528213629722.png)



## SimPO

DPO采用隐式奖励函数的缺点：

（1）需要ref模型，带来额外内存、计算成本

（2）训练阶段的奖励函数

![image-20240529100307738](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240529100307738.png)

和推理阶段的生成指标（最大化平均对数似然）存在差异，

![image-20240529100344613](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240529100344613.png)

### 推导

直接用（3）式作为奖励函数：

![image-20240529100858564](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240529100858564.png)

引入奖励差额：

![image-20240529101126325](C:\Users\viruser.v-desktop\AppData\Roaming\Typora\typora-user-images\image-20240529101126325.png)