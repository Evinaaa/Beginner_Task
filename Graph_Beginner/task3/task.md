# task3

## 数据集的选择

- 我们这次选择的数据集是TUDatasets，这是TU Dortumnd University收集的大量关于分子特征的图数据
- 获取：数据集TUDatasets可以通过[PyTorch Geometric](https://zhida.zhihu.com/search?content_id=238708319&content_type=Article&match_order=1&q=PyTorch+Geometric&zhida_source=entity)直接加载
- 这个数据集有188张图，有两个类。通过查看第一个图的基本信息，我们可以看到它有**17个节点**、**38条无向边，**还有一个图的标签`y=[1]`表示图是哪一类的（1维向量，一个数）。

## 实验流程

加载数据——构建模型——创建训练集和测试集(本次实验中训练集和测试集的比例为7:3）——输出结果

## 实验结果

- 训练过程输出如下所示，可以看到在训练集上准确率可以达到 `75%`，在测试集上达到了约 `74%` 的准确率：