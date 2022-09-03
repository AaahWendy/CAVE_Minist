# CVAE_MNIST
使用MNIST数据集进行训练

进行二十次循环后，trainloss和validloss函数图像如下： 

![](https://github.com/AaahWendy/CVAE_MNIST/blob/master/fig/4.png)

（在更多次训练中发现，收敛效果并不十分理想）

在训练结束后、随机抽取十个标签，生成的图片如下图：

第一行为原始图片，

第二行是根据原始图片生成的自编码数字，

第三行为根据原始图片的label生成的模拟数据  
![](https://github.com/AaahWendy/CVAE_MNIST/blob/master/fig/2.png)
![](https://github.com/AaahWendy/CVAE_MNIST/blob/master/fig/myplot.png)
![](https://github.com/AaahWendy/CVAE_MNIST/blob/master/fig/myplot2.png)
