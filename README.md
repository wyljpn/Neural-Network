# Neural Network
AClassification Problem

1.SYSTEM: ATwo-Nested-Spirals Problem

Two-Nest-Spiralsproblem is a well-known classification benchmark problem. It contains twonested spirals, ‘o’ and ‘+’, as shown in figure. The task is to separate thetwo nested spirals.

用matlab实现BP算法，其中加入了L2正则化项，在隐藏层最后一层使用Dropout。输出进行独热编码（one-hot），使用softmax。在图上打印出模型进行判断的边界。

 有4个文件。

1）TwoNestSpiralsUseGivenSet.m

2）ReLU.m

3)  ReLUGradient.m

4)  softmax.m

应该要把定义一个前向传播的函数。应该在训练和测试的时候都要用到。如果修改了训练中前向传播的代码，而忘记修改测试的代码，则会出错或者产生奇怪的结果。但是感觉matlab中传递权重和偏置给函数中很麻烦，貌似不能直接将所有权重和偏置放入一个集合，然后在函数中重新取出。

