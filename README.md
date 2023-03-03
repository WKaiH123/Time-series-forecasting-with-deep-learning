# Time series forecasting with deep learning
基于深度学习的时间序列预测工具包（测试版）


## 欢迎大家使用和指正！！！ （Welcome to use and correct！！！）   
### Email：wkaihan168@gmail.com

## 每个模型的参数范围需自行针对每个任务进行合理的设定。

### 函数或类的功能简介（A brief description of the function or class）

| Name      | Description | Type     |   Other  |
| :----:        |    :----:   |    :----:   |    :----:   |
| calculate      |  进行回归预测任务的评估指标  |  function  |    |
| load_data   |  导入实验数据集，并划分训练集和测试集，并进行标准化,最后批量化数据输出 | function  |    |
| RNNModel   |  RNN模型结构  |  class  |     |
| MLPRegress   |  MLP模型结构  |  class  |     |
| LSTM   |  LSTM模型结构  |  class  |     |
| TCN   |  TCN模型结构  |  class  |  包括Chom1d,TemporalBlock,TemporalConvNet等组件   |
| GRU   |  GRU模型结构  |  class  |     |
| BLSregressor   |  封装好的 Bord Learning System 的回归模型类  | class  |     |
| models   |  训练机器学习和深度学习模型的类  |  class |   各个模型的网络层数，学习率，优化器要根据具体任务具体设置   |

### 模型测试结果将自动保存在以下文件夹中（Model test results）：
| File_name      | Description |
| :----:        |    :----:   |
| figure        | 保存模型测试集的拟合图|
| models        | 保存最佳模型         |
| prediction    | 记录测试集的预测值与真实值  |
| results    | 记录模型训练过程中的评估结果  |
