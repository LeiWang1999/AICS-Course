选上陈云霁老师的课之后，开始更新课程配套的实验。

部分代码参考自：[AI-homework](https://github.com/LuoXukun/AI-homework)

### Exp_2_1

Socre: 100

Accuracy: 0.981（在训练过程中加了一个简单的save best model)

Hidden Size: 100,100

Epoches: 20

作业递交评分：

| Experiment Name | Score |
| --------------- | ----- |
| Exp_2_1         | 100   |
| Exp_2_2         |       |
|                 |       |

### Exp_2_2

Socre: 100

Accuracy: 0.981

Hidden Size: 100,100

Epoches: 20

![image-20220311140145079](https://leiblog-imgbed.oss-cn-beijing.aliyuncs.com/img/image-20220311140145079.png)

满分的要求是让DLP跑的能比CPU快50倍，这个服务器用这么好的CPU干神魔！跑的贼快，目测是numpy的矩阵乘法上了CPU的向量指令,这样一通优化下来速度还真不一定比DLA慢多少,,于是这里我对forward的代码进行了小的改动，把原来numpy高效的dot product的batch纬度展开（小小的多一层循环：

```python
        # TODO：全连接层的前向传播，计算输出结果
        """ y = X * W + b """
        self.output = np.zeros((batch_size, self.num_output))
        #self.output = np.matmul(self.input, self.weight) + self.bias
        for n in range(batch_size):
            self.output[n] = np.matmul(self.input[n], self.weight) + self.bias[0]
```

果然速度慢了无数倍，为了不卡太久，手动把batch_size调低到10！

（话说直接在cpu推断这部代码加上sleep 1s算了XDD)