## 北理工数据挖掘大作业 - 微博推荐item预测

> 作业提交时间：2016.7

### 学生信息
- 宁小东 2120151024
- 黄建峰 2120150994
- 王新灵 2120151042

### 数据集

含有4.08GB微博数据，包含用户资料、SNS等。Github中上传的版本摘取了部分数据，包含训练集和测试集两部分。训练集含30000条数据，测试集含10000条

### 处理步骤

数据的处理步骤如下：  
  1. 对原数据进行清洗和合并，获取适合训练-测试的数据集。  
  2. 对训练集进行逻辑回归分析，将获取的回归函数应用于测试集，获取推荐结果。  
  4. 将最终结果进行可视化分析，得出结论。  
  3. 最终报告请详见Final_report.pdf。  

### 包含文件

- costFunction.m 计算逻辑回归的损失函数 
- costFunctionReg.m 带正规化的costFuncion函数 
- log_norm.m log-正态分布函数，用于计算sigmoid 
- mapFeature.m 备用的特征映射函数。当增加训练集每条信息的特征维度时，可应用本函数降维 
- plotData.m 备用的可视化函数。将一组X和y数据进行可视化，本程序中是将测试集作为X，预测结果作为y 
- plotDecisionBoundary.m 同上，添加了精度确界theta 
- predict.m 用于预测的函数，调用了sigmoid函数 
- sigmoid.m 逻辑回归中的sigmoid函数 
- weibo_predict.m 主函数，运行获取最终的推荐item列表 
- test_log_demo.csv 合并后的测试数据集，每行数据包含18维特征 
- train_log_demo.csv 合并后的训练数据集，同上 
- test_full_y.csv 预测的推荐item列表（__实验结果__），每行对应一个test_log_demo中的用户 
