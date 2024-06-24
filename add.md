#### PPO
1.标注数据训练一个sft model
2.pair打分数据训练reward model（计算即时收益）
3.ppo方式训练policymodel（目标LM）, reference model（加上约束，防止寻歪了）和critic model（来自RM，预估总收益）
   3.1 初始化参数θ和θ'（off-policy策略，交互和学习是不同步的，θ'负责交互产生数据来更新θ，这样更节省资源和时间）
   3.2 用θ'与环境交互产生状态行为数据（st,at），计算advantage Aθ'
   3.3 使用ppo目标函数来拟合θ（奖励部分：*importance weight和优势* 和 KL部分）

1.采样
actor和critic，根据prompt生成结果，包括：
response N个token；N个token的对数概率；N个预估收益
2.反馈
rewardmodel 对response打分，即时收益，注意是在最后一个token输出后计算的。
同时算referencemodel对N个token的对数概率 减去 actor的对数概率，防止偏移，KL散度正则项的作用
3.学习
优势=实际收益 - 预期收益
实际收益表示从当前token到最后一个token的奖励总和，预期收益就是critic的输出
(直觉理解)实际收益都是正值，每个动作都是正向的是不对的，所以要加一个base基础。
actor的loss表示为优势a 和p(token)的乘积
critic的loss 均方误差（预期收益 - 实际收益）的平方
最终通过拟合两个loss的加权和，来更新。


Agent
Thought-Action-Observation