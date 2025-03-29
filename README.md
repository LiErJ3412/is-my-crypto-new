# CTF密码学原题检索系统
一个基于语义搜索的CTF密码学题目相似性检索引擎。
根据<a href="http://yuantiji.ac" target="_blank" style="color: blue">http://yuantiji.ac</a> 改编而来。Orz zzq大佬。

同时感谢Copilot辅助编程（qwq

**更新 (2025/3/29):** 数据收集完成！基于语义搜索构建了CTF密码学原题库，收录了来自各大CTF比赛的密码学题目。使用 **DeepSeek-R1-Distill-Qwen-32B**作为LLM模型，通过 [硅基流动](https://cloud.siliconflow.cn/) 提供服务，搭配 
**bge-large-en-v1.5** 进行向量嵌入。

#### 工作原理

这个项目的核心思路与原题机相似：

1. 利用LLM对CTF密码学题目进行标准化处理，提取核心的密码学问题并去除变量名的影响
2. 对处理后的题目描述和查询进行向量嵌入，实现语义相似性搜索
3. 快速检索出与当前题目相似的历史CTF密码学题目，帮助识别"换皮题"（但识别率并不是很高;_;）...
(例如ciscn2025初赛的fffffhash,与DUCTF2023的fnv相似,但仅仅将mod操作换了表达方式就无法识别了) 

#### 部署指南

你需要注册硅基流动去获取API密钥，可以在官网查看价格详情，自行去更换模型（。

将CTF密码学题目放在`ctfproblem/`文件夹中，按照示例格式（`ctfproblem/1001.json`）命名。命名可以是任意的，也可以使用嵌套文件夹。运行以下命令：

1. `python -m src.build_summary` 获取题目的简化描述
2. `python -m src.embedder` 构建向量嵌入
3. `python -m src.build_locale` 检测题目语言
4. `python -m src.ui` 启动服务

对于大规模运行，建议使用较好的CPU，因为向量搜索是CPU密集型操作。你可能需要在`src/ui.py`中修改`max_workers`参数调整并发量。

#### 数据说明

数据库目前包含了600+场比赛的2000+道CTF密码学题目，覆盖常见的密码学类型，包括：经典密码、现代密码、公钥密码、哈希、随机数生成等领域。每个题目包含JSON格式的元数据和VOPKL格式的向量表示。碍于侵权问题，库中不会直接提供赛题数据。

**效果可能不是很好，请多多包涵;_;喜欢的话就点个star吧QAQ**