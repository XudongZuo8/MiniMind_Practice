### 1.框架图
![alt text](Users/xudongzuo/GitHub/XudongZuo8.github.io/images/note/image-1.png)
![alt text](Users/xudongzuo/GitHub/XudongZuo8.github.io/images/note/image-2.png)
上面两张图，其一是MiniMind的框架图，其二是MoE的图，两个相差不大，区别在于用MoE替代了基础版LLM的FFN模块。
Rope旋转位置编码


项目结构
RMSNorm
首先为什么需要Norm（也就是归一化，均值为0，标准差为1）

数据集：
![alt text](Users/xudongzuo/GitHub/XudongZuo8.github.io/images/note/image.png)
这部分在官方的文档里写得清楚，同时发布于魔塔社区和huggingface。
