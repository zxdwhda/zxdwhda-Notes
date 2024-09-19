---
UID: 20240902132632 
aliases: 
tags: 
source: 
cssclass: 
created: 2024-09-02
---

## ✍内容
Transformer 模型通过处理长文本序列和比传统 RNN 和 LSTM 更有效地理解上下文，彻底改变了自然语言处理 (NLP) 领域。该架构基于自注意力机制，允许模型在进行预测时权衡句子中不同单词的重要性。

应用：文本生成、翻译、问答等。

**行业用例：OpenAI 的 GPT-3**

OpenAI 的 GPT-3 是一种 Transformer 模型，可以根据给定的提示生成连贯且上下文相关的文本。它可用于聊天机器人、内容创建和编程辅助。

对各行各业的好处：通过利用类似人类的文本生成功能，实现客户服务自动化、增强内容创建并降低运营成本。





# 相关文章摘抄一

https://www.zhihu.com/question/589738603/answer/3415135020

Transformer 的优势吧，

- 被广泛证明过，从最早的 Bert 开始Transformer 的 Encoder 与 Decoder 架构就证明它在大算力下的优势：并行且可以有序列数据的相关性。但是很遗憾，最先发现并且不遗余力的推行“规模就是智能”这事的，结果大家看到了 ChatGPT 、 GPT4、 DALL-E 、 SORA、 Whipser这些都是指导思想下的产物。
- 可扩展性，很多时候你只要找到了一个稳定的方式，你就可以简单的堆模型的单层并行变大，或者把模型加几层，那你的效果就会持续变好。而这个特性，CNN 、 RNN 好像没有那么容易，还记得 CNN 时代，从几层到十几层，到 100 多层它的架构基本上都不太一样，这是不太适合工作生产的。
- 特别适合显卡算力，显卡本身是一个完全异于 CPU 计算的算力设备，它适合大规模的并行计算，但是不适合复杂运行。比如H100， 这里边你能看到的小单元都是并行 Core。如果我们常见的 CPU 有几十个核已经算是最优秀的版本，那 Nvidia 的显卡至少要有 16896 个FP32 核 才是个 H100 的标配。这样的东西运行起 Transformer 确实是绝配！
- 结果是端到端的，从输入到输出的整个过程都是可以在一个统一的框架内完成，简化了模型设计与训练。特别受欢迎的一种工作方式！
- 适应性广，过去的 CNN 、 RNN 都是针对单一数据的，而这次的 Transformer 架构，已经证明了在 自然语言/文本数据、序列数据、图像、语音、视频等各类数据，各种任务上都有了极其优异的表现，基本上都是 SOTA 级别的，你说它能一统天下都是对的。那大家还有什么动力去研究别的架构呢？你即使再优秀的架构，如果赶不上时间窗口，那还有什么意义？

不过必须说回来，现在还是有很多做学术的人在研究不同的新的架构。

最近的 Mamba、国内比它早的 RWKV 都是非常优秀的基于 RNN 的大语言模型的架构，但是并没有得到大规模的推广。不过我们可以简单的看一下这两个架构吧。

**Mamba ：**

Mamba结合了[状态空间模型](https://zhida.zhihu.com/search?q=%E7%8A%B6%E6%80%81%E7%A9%BA%E9%97%B4%E6%A8%A1%E5%9E%8B&zhida_source=entity&is_preview=1)（State Space Model, SSM）的特点，旨在提供线性[时间复杂度](https://zhida.zhihu.com/search?q=%E6%97%B6%E9%97%B4%E5%A4%8D%E6%9D%82%E5%BA%A6&zhida_source=entity&is_preview=1)的序列建模。Mamba的核心创新在于引入了选择性状态空间（Selective State Space Model），这使得模型能够根据输入内容选择性地关注或忽略特定的信息。

核心架构如下

![[99-Attachment/Pasted image 20240902133007.png]]

**RWKV ：**

RWKV（Receptance Weighted Key Value）架构是一种结合了RNN（[循环神经网络](https://zhida.zhihu.com/search?q=%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C&zhida_source=entity&is_preview=1)）和Transformer优点的新型网络。它旨在实现高度并行化训练与高效推理，同时保持时间复杂度为线性复杂度，特别是在长序列推理场景下。

**核心架构如下**

![[99-Attachment/Pasted image 20240902133031.png]]

而 Transformer 的架构如下：
![[99-Attachment/Pasted image 20240902133053.png]]

可以看到好像都挺复杂，但是当 Transformer 只剩下 Decoder，那它的结构就比上面两个要简单一点儿，而且都是并行计算单元了，更适应现在的算力方式。

# 相关文章摘抄二
#### **Transformer架构**

当下最火的当属2017年的transformer架构，Transformer是目前最常见的语言模型的基本结构。transformer架构涉及大量的概念和应用，比如编码-解码（encoder-decoder），注意力机制（attention），kqv（key、Querry、value）等。

Transformer模型的核心架构可分为编码器和解码器。即编码器将输入序列编码成一个向量，而解码器则从该向量中生成输出序列。

简单的工作流程如下：

![图片](https://mmbiz.qpic.cn/mmbiz_png/33P2FdAnju9eLjuOQfgf4qEpwWSrPace6PiaR1SmJ7wbCqgicFbuY6J9jImwNqf88g1ibibTPVXiaQS78KPk8Rjl7HQ/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1&wx_co=1)

1、获取输入句子的每一个单词的表示向量 X，X由单词的 Embedding（Embedding就是从原始数据提取出来Feature） 和单词位置的 Embedding 相加得到。

![图片](https://mmbiz.qpic.cn/mmbiz_png/33P2FdAnju9eLjuOQfgf4qEpwWSrPaceAj2rcttru0kYDWSmOIKBme4fCTD3kTv9PW3ADXU1f2JSHFte6sd6hw/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1&wx_co=1)

2、将得到的单词表示向量矩阵 (如上图所示，每一行是一个单词的表示 x) 传入 Encoder 中，经过6个 Encoder block 后可以得到句子所有单词的编码信息矩阵C

3、将 Encoder 输出的编码信息矩阵 C传递到 Decoder 中，Decoder 依次会根据当前翻译过的单词 1~ i 翻译下一个单词 i+1 。

![图片](https://mmbiz.qpic.cn/mmbiz_png/33P2FdAnju9eLjuOQfgf4qEpwWSrPaceGiaiaHFpudHPhxIa9q3vbj5ibOib49fmstHsIcBmU0E7SMbQCDn79RwrKQ/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1&wx_co=1)

使用ChatGPT的时候会发现输出结果是一个字一个字蹦出来的，这是Transformer的结构导致的。

简单理解：可以将Transformer模型学习和预测的过程看成是语言翻译。如果模型是将A语言翻译成B语言，那么Transformer模型结构中的编码器是将输入的A语言翻译成模型语言，而解码器则是将模型语言翻译成B语言。

#### **▐**  **注意力机制**

Transformer模型之所以具备强大的功能，可以归功于模型中应用的注意力机制。何为注意力机制？对于一张图，我们并不会同等地查看图中的每个位置，而会自动提取“重要的位置”。

Attention = 注意力，从两个不同的主体开始。（两个主体互相注意，我注意到他，他注意到我）

NLP领域最开始用于翻译任务，天然是source、target，D翻译第一个词的时候，有个attention的机制关注到前面的所有词，但是权重不一样。简单理解：**计算词之间的相近关系**。

![图片](https://mmbiz.qpic.cn/mmbiz_png/33P2FdAnju9eLjuOQfgf4qEpwWSrPaceInvuESPBNBB9nGhhmKibP72Ucq0aumBb0GnKqkkwibUcN7Jw5MfY726w/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1&wx_co=1)

注：颜色粗细代表权重大小

以上图片可以解读为：**一段自然语言内容，其自身就「暗含」很多内部关联信息**。例如上面这句话，如果用“自注意力”机制，应该给与“知识”最多的注意力，因此可以认为：

**一段自然语言中，其实暗含了：为了得到关于某方面信息 Q，可以通过关注某些信息 K，进而得到某些信息（V）作为结果**。（Q 就是 query 检索/查询，K、V 分别是 key、value。所以类似于我们在图书检索系统里搜索「NLP书籍」（这是 Q），得到了一本叫《自然语言处理实战》的电子书，书名就是 key，这本电子书就是 value。只是对于自然语言的理解，我们认为任何一段内容里，都自身暗含了很多潜在 Q-K-V 的关联。）【qkv机制后续在图片领域也有大量的应用，可以熟悉一下这个机制】

关于transformer架构，还有很多的逻辑和知识，不做枚举。且后续大量的逻辑会基于向量和矩阵展开，不易理解。简单的罗列下为什么这个架构后面带来了大量的变革。即架构的优势：

- 快：比起2017年前的rnn，transformer并行性更好；
    
- 记忆力好：词间距缩短为1，长文本的时候，可以有更多的容量；
    
- 处理不同长度的序列：不需要输入的数据序列是固定长度的。

#### **▐**  **上下文学习（In-Context Learning）**

一个预训练模型，在处理下游任务时，不微调模型参数，只需要在输入时加一些示例，就能有 SOTA（state-of-the-art，即最优秀的模型） 的表现，这就是模型的上下文学习（In-Context Learning，ICL）能力。

1. ICL 能力的直接应用：Prompt Engineering
1. 2022 期间很多学界人士的研究重点都转向了 Prompt。首先一般性地「Pretrain, Prompt」到了 Prompt 环节，可能是给模型输入 x 期望得到输出 y。但是如果我们对使用者给出的 x 进行二次加工（比如把这个加工表示为一个函数 f（x)），是否能在输出上获得更好的结果 y 呢？甚至可以优化输出的结构，得到更好的结果。
2. 举个例子。比如模型的使用者想问「自驾去杭州周边两天一夜玩，有什么推荐的地方吗？」，模型返回了「南浔古镇」。而如果通过 Prompt Engineering 优化一下可以这样：
![图片](https://mmbiz.qpic.cn/mmbiz_png/33P2FdAnju9eLjuOQfgf4qEpwWSrPaceKItQha8nXTBbbNpOLnXiajp0JL28NeTroNF8vZksFfg8RZtvWRlIuIA/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1&wx_co=1)

这样 f(x) 就是 Prompt Engineering，而 g(x) 其实是 Answer Engineering。

2. ICL的数学原理和底层逻辑其实目前没有明确定论，也比较复杂。简单对ICL总结用于指导后续应用，包括：
- 在 prompt 里带上 demo 是很重要的，而且 demo 在形式上 input 和 label 都需要。
- 对于 demo 中的 input，不要乱来，要给出比较合理的 input。
- 对于 demo 中的 label，只要它属于正确的值域空间 label space 就可以了，是否与 input 有 correct mapping 不重要。
#### **▐**  **Prompt Framework**

  

Prompt的专业与否 直接关系到以下两方面的结果效果：

1、大模型回复的准确性和针对性：好的Prompt才可以帮助模型更好地理解你的意图和需要。

2、大模型回复语句的自然度：好的Prompt可以帮助模型更好地处理歧义，以及上下文依赖性等问题，提高模型回答的自然度。

  

Elavis Saravia 总结的框架：

- Instruction（必须）：指令，即你希望模型执行的具体任务。
    
- Context（选填）：背景信息，或者说是上下文信息，这可以引导模型做出更好的反应。
    
- Input Data（选填）：输入数据，告知模型需要处理的数据。
    
- Output Indicator（选填）：输出指示器，告知模型我们要输出的类型或格式。
    

  

只要你按照这个框架写 prompt ，模型返回的结果都不会差。

  

当然，你在写 prompt 的时候，并不一定要包含所有4个元素，而是可以根据自己的需求排列组合。比如拿前面的几个场景作为例子：

- 推理：Instruction + Context + Input Data
    
- 信息提取：Instruction + Context + Input Data + Output Indicator
    

  

#### **▐**  **提示工程（Prompt Engineering）**

  

在Prompt Framework背景下，为了让LLM有更高质量的回答、对业务有价值的回答。在工程实践中，工程侧会将用户的随意问题进行包装组织以及抽取，组成问题的上下文，并添加一些周边的限定语句，而得到有效提示词的代码逻辑或者应用服务（有效提示词组织服务：能完成这些复杂应用业务的 一般是需要一个专门的服务）。

  

query有效->产出好的结果是不容易 -> 避免二义性、无结果

所以产生了提示词工程：提示词引擎-runtime(串联产投链路数据) & 模板运维

- 分离：模板：模板与工程分离 （类似前后端分离）
    
- 信息够不够：串联产投链路数据，补齐到模板中
    

  

#### **▐**  **C****OT(chain of thought) 思维链**

  

深度学习的演变阶段：在认知科学里，有一个「认知双通道理论」，讲的是人脑有两套系统，即「系统 1」和「系统 2」：

- 系统 1（System-1）常被称为直觉系统，它的运行是无意识且快速的，不怎么费脑力，没有感觉，完全处于自主控制状态。
    
- 系统 2（System-2）常被称为逻辑分析系统，它将注意力转移到需要费脑力的大脑活动上来，例如复杂的运算。系统 2 的运行通常与行为、选择和专注等主观体验相关联。
    
      
    

System-1 是目前深度学习正在做的事情 —— Current DL，比如图像识别、人脸识别、机器翻译、情感分类、语音识别、自动驾驶等。System-2 是未来深度学习将要做的事情 —— Future DL，比如推理、规划等任务，这些任务基本都是有逻辑的（logical）、可推理的（reasoning）。

  

大语言模型研究者们也在探究那些 System-2 要解决的任务，于是有了下面的几个技术方向。

1. **Google 提出思维链提****示（CoT Prompting）**：2022年初google在论文里面提出「思维链（Chain of Though，CoT）」，研究发现通过在 prompts 中增加思维链（即一系列中间推理步骤），就能显著提升 LLM 的推力表现。并将这种提示方式称为「Chain of Thought prompting」。
    
    标准的提示中给了一对问答样例，再加上一个问题，让 LLM 返回问题的答案。CoT Prompting 则在那对问答样例中加上了一段 CoT，而不是直接给出「The answer is 11.」下图展示了这样两种 prompting 及对应输出的案例。
    
      
    
    ![图片](https://mmbiz.qpic.cn/mmbiz_png/33P2FdAnju9eLjuOQfgf4qEpwWSrPaceUCdxRF2iawp7khxOIvv4sOpKwAXFZqTB4hbjkN83lXMoU0ar4o8NKicA/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1&wx_co=1)
    
2. **Let's Think Step by Step** 
    
    2022 年 5 月三位东京大学学者与两位 Google 的研究人员中提到了后来在 Gen-AI 领域那句著名的提示 —— Let's think step by step —— 对于涉及到逻辑推理方面的问题，通过增加这句提示后，模型展现出了推理性能的大幅跃升。  
    

  

对于 OpenAI 的 InstructGPT（具体地，是 text-davinci-002）模型，在输入提示时加上「Let's think step by step」后，其表现：

- 在 MultiArith 数据集上，准确率从 17.7% 提升到 78.7%
    
- 在 GSM8K 数据集上，准确率从 10.4% 提升到 40.7%
    

  

对于 Google 的 PaLM 模型（具体地，参数规模为 5400 亿），同样的输入提示改造，其表现：

- 在 MultiArith 数据集上，准确率从 25.5% 提升到 66.1%
    
- 在 GSM8K 数据集上，准确率从 12.5% 提升到 43.0%
    

  

事实上，还有很多类似的Prompt插入，但是效果不如这句好。