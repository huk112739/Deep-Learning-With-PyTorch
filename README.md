# Pytorch 深度学习实战

![Deep Learning With PyTorch](https://github.com/huk112739/Deep-Learning-With-PyTorch/blob/main/dataset/photo.png)

本项目是对《Pytorch 深度学习实战》源码复现。书中示例源码已公开在[
DLWPT-code](https://github.com/deep-learning-with-pytorch/dlwpt-code.git)仓库中。为了更好的学习和编写示例，对示例代码中part1中部分变量单独显示，并增加了一些torch API的练习。在part2中为了适配当前版本的环境，修改部分包的引用。

***
### 项目结构
&emsp;|--data_out<br />
&emsp;&emsp;|--chapter *<br />
&emsp;|--dataset<br />
&emsp;&emsp;|--chapter *<br />
&emsp;&emsp;|--part2<br />
&emsp;&emsp;&emsp;|--data_process<br />
&emsp;&emsp;&emsp;&emsp;|--dataset<br />
&emsp;&emsp;&emsp;&emsp;|--util<br />
&emsp;|--part2<br />
&emsp;|--chapter *.ipynb<br />

PS：项目练习在part2中
### 修改
- 书中源代码 p2ch10.dsets.py中，第37、86行分别修改为：<br />
```python 
[37]  mhd_list = glob.glob('dataset/part2/luna/subset/*.mhd')
```
```python 
[86]  mhd_path = glob.glob('dataset/part2/luna/subset/{}.mhd'.format(series_uid))[0]
```
## 运行
#### 1.下载数据集
下载[LUNA16](https://luna16.grand-challenge.org/Download/)数据集，国内可以从百度网盘下载。
```
百度云地址:
    链接：https://pan.baidu.com/s/1q6piaUc867e1CcZgKUzfNg?pwd=9vtu 
    提取码：9vtu 
```
#### 2.安装环境
(1)创建虚拟环境,并激活今日虚拟环境
```shell
conda env -n deep_learning python=3.9
conda activate deep_learning
```
(2)快速导入环境包
```shell
pip install -r requirements.txt
```