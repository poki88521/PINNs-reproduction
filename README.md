## 简介
PINNs（物理信息神经网络）有关论文复现（代码、结果与总结）仓库

## 说明
- 本仓库主要包含复现的PINNs论文代码、图表
- 这个仓库经历过一些大改，目前复现代码的初版已经迁移到[这个项目](https://github.com/poki88521/PINN-bench)了
- csv数据表格太多太大了，用gitignore屏蔽了（只保留代码、图和模型权重）

## 内容总结
### 1.Hidden Fluid Mechanics
论文引用：`Maziar R ,Alireza Y ,Em G K .Hidden fluid mechanics: Learning velocity and pressure fields from flow visualizations.[J].Science (New York, N.Y.),2020,367(6481):1026-1030.DOI:10.1126/science.aaw4741. `
- 本论文是PINNs的开篇论文之一，不涉及特殊网络结构，只作为代码复现练习，故暂无特殊总结
- 但本论文的复现还是涉及很多PINNs特有的代码设计，包括：
    - 损失函数：不使用ReLU而使用二阶导平滑的Swish
    - 权重归一化而非批次归一化：防止样本间依赖
    - 网络深度通常较大来拟合复杂函数
    - 物理信息嵌入部分采用`torch.autograd.grad`自动计算，同时必须设置`create_graph=True`否则计算不了二阶导
    - 计算数据残差和计算物理信息残差的数据集要分别抽取，来保证尽量不重合
    - 采样时时间头尾包括边界条件，需要保留且固定
    - PINNs可能不存在完整的数据集（或数据集的完整性无意义），所以迭代数iterations是一个重要的衡量指标
    - 测试点选取绕流稳定的时间中段
- 总结：开山论文的深化（但还是比2019年那个细化了一些）

### 2.Improved-PINN
论文引用：`Niu P ,Guo J ,Chen Y , et al.Improved physics-informed neural network in mitigating gradient-related failures[J].Neurocomputing,2025,638130167-130167.DOI:10.1016/J.NEUCOM.2025.130167.`
- 一篇很多ai都推荐的论文
- 加了自适应权重的损失函数：
$$
\mathcal{L}(\theta,\boldsymbol{\sigma}) = 
\frac{1}{\sigma_{ic}^2+\gamma^{-1}}\mathcal{L}_{ic}(\theta) + 
\frac{1}{\sigma_{bc}^2+\gamma^{-1}}\mathcal{L}_{bc}(\theta) + 
\frac{1}{\sigma_r^2+\gamma^{-1}}\mathcal{L}_r(\theta)
$$
$$
 + 
\log(\sigma_{ic}^2+\gamma^{-1}) + 
\log(\sigma_{bc}^2+\gamma^{-1}) + 
\log(\sigma_r^2+\gamma^{-1})
$$
- 注意力网络：做了两个可学习编码器U、V，把初始数据输入隐藏层，解决梯度消失的同时还可以让网络自动调节关注点
- 收敛的很快而且没有震荡，很稳定

### 3.Scale-PINN
论文引用：`Chiu, P.-H., Wong, J.C., Ooi, C.C., Wei, C., Fan, Y., Ong, Y.-S., 2026. Scale-PINN: Learning Efficient Physics-Informed Neural Networks Through Sequential Correction. https://doi.org/10.48550/arXiv.2602.19475`
- 目前为止最新的一篇论文，而且成果看起来很突出（代表性问题2分钟以内，精度提高）
- 加了一个平滑算子
$$\mathcal{P}_{\alpha} = (I - \alpha^{2} \nabla^{2})$$


## 日志
- 跳转链接点[这里](log.md)
- 论文复现（不完整）日志，记录当日的操作和更新内容，按日期进行分隔（日期不连续）
- 以下为摘要&重点（由deepseek总结）：


## 笔记
- 跳转链接点[这里](note.md)
- pinn复现过程中的日记，内容包括但不限于知识笔记、收获总结、问题提出、碎碎念等，较为杂乱，按日期分隔
- 以下为摘要&重点（由deepseek总结）：
