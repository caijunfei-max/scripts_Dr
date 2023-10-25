<font size = 30 face = "微软雅黑">Python的库与函数——Cai Junfei个人笔记</font>

[toc]




# 特殊函数

## assert
```python
assert expression
```
相当于
```python
if not expression
    raise AssertionError
```
也就是说，如果没有assert后面的条件判断为False，则会输出AsserttionError异常



常用内置库
## optparse(设计命令行)

创建命令实例：每个parser实例代表一类命令实例。 例如-f是文件命令中的一个option，它属于文件parser对象；-zxvf是压缩解压命令中的一个option，它属于压缩解压parser对象。

```python
from optparser import OptionParser
parser = OptionParser()
```
详尽介绍见[optparse官方手册](https://docs.python.org/3/library/optparse.html#optparse-standard-option-actions)

### OptionParser.add_option()

设置parser的行为的：例如
```python

parser.add_option("-f", "--file", action="store", type="string", dest="filename")
```
其中-f和--file都是选项，前者是缩写，后置是全写。action、type、dest等都是解析器对象的重要属性，详见optparse官方手册

# python官方库
## os(Miscellaneous operating system interfaces)
### os.mkdir 
简而言之，创建文件夹的，直接输入路径就行了，其中包含创建文件的文件名。其他参数见官方文档

### os.rename重命名文件
两个参数，第一个参数是准备被改动的文件，第二个参数是是目标名字

### os.walk检索目录下所有文件
也可以使用os.listdir
参数是路径，返回的是三个值，分别是根目录，子文件，和文件名
代码如下：
```python
def scan_files(directory, prefix=None):
    files_list = []

    for root, sub_dirs, files in os.walk(directory):
        for special_file in files:
            if prefix:
                if special_file.startswith(prefix):
                    files_list.append(special_file)
            else:
                files_list.append(special_file)

    return files_list
```
附：这里的startswith是判断字符串前缀的，如果判断字符串后缀用endswith

## shutil（高级文件处理High-level file operations）
用于文件编辑的高级操作
### shutil.move移动文件
常用两个参数，第一个参数事你要移动的文件，第二个参数是目标目录（路径名）。文件不存在或者是目录不存在，重复都会报错。

## 系统sys

### sys.argv
获取外部参数
例如，一个test.py程序运行时有五个参数：test.py a b c d e
那么在test.py中加入以下不同内容会得到不同内容：
a = sys.argv[0] --> 索引为0的，为test.py
a = sys.argv[1] --> 索引为1的，为a
a = sys.argv[2:] --> 索引为2之后的，为['b','c','d','e']

因此sys.argv本质时一个获取了一个列表，其中的元素是程序本身和相关参数

## 时间time
### time.sleep休眠
sleep() 方法暂停给定<font color = red>秒</font>数后执行程序。该参数可以是一个浮点数来表示一个更精确的睡眠时间

```python
time.sleep(t)#t是要暂停执行的秒数
```
实例

```python
import time

print ("start")
time.sleep( 5 )
print ("5 seconds after")

```

### time.time
返回当前时间时间戳（1970纪元后经过的浮点秒数）

### time.localtime格式化时间戳为当地时间

```python
import time

localtime= time.localtime(time.time())
```

## 数学math
### math.sqrt
返回数字x的平方根

```python
a= math.sqrt(x)
```


## 字符re
### re.compile

编译正则表达式返回正则表达式对象，两个参数

```python
re.complile(pattern, flag=0)
```
pattern是正则表达式（字符串形式），flag是一个可选 参数，规定匹配模式，比如忽略大小写，多行模式等等

### re.match

用正则表达式匹配字符串，若成功，返回匹配对象，否则返回None

    re.match(pattern,string,flags=0)

判断一个字符串（string）是否满足一个正则化表达式（pattern）

### re.search
搜索字符串中第一次出现正则表达式的模式 成功则返回匹配对象 否则返回None
    
    re.search(pattern, string, flags=0)

### re.split

**split(pattern, string, maxsplit=0, flags=0)**	用正则表达式指定的模式分隔符拆分字符串 返回列表

<details>
<summary>拆分长字符串</summary>

```python
import re


def main():
    poem = '窗前明月光，疑是地上霜。举头望明月，低头思故乡。'
    sentence_list = re.split(r'[，。, .]', poem)
    while '' in sentence_list:
        sentence_list.remove('')
    print(sentence_list)  # ['窗前明月光', '疑是地上霜', '举头望明月', '低头思故乡']


if __name__ == '__main__':
    main()
```

</details>

### re.sub

**sub(pattern, repl, string, count=0, flags=0)**	用指定的字符串替换原字符串中与正则表达式匹配的模式 可以用count指定替换的次数

<details>
<summary>

### re.fullmatch

**re.fullmatch(pattern, string, flags=0)**	match函数的完全匹配（从字符串开头到结尾）版本

### re.findall
**re.findall(pattern, string, flags=0)**	查找字符串所有与正则表达式匹配的模式 返回字符串的列表

### re.finditer
**re.finditer(pattern, string, flags=0)**	查找字符串所有与正则表达式匹配的模式 返回一个迭代器

# 图像处理库

## pillow

### 

# 网络爬虫

## request

[官方文档](https://docs.python-requests.org/zh_CN/latest/)

# Numpy
## 数组生成
### numpy.loadtxt

从文本文件加载数据，
这里文本文件里面的数据每一行应该具有相同数量的值才可以成功加载。加载时会将文本中的第一行数值数作为设定的数量，下面哪一行出错都会显示如下的报错：“<font color = red>ValueError: Wrong number of columns at line </font>”

在对这样的数据进行加载时
```
1 2 3 4 5 2
3 4 5 6 7 8
2 4 5 8 4 4
```
使用以下的代码进行加载
```python
import numpy as np
a = np.loadtxt('abc.txt',dtype=bytes)
print(a)
```

结果如下
```
[[1. 2. 3. 4. 5. 2.]
 [3. 4. 5. 6. 7. 8.]
 [2. 4. 5. 8. 4. 4.]]
```

为了方便处理可以通过<font color=blue>astype</font>方法将数据转化成其他类型，dtype转化为bytes方便读取非数字的数据，比如字符串
如下：
```python
a = np.loadtxt('abc.txt',dtype=bytes)
print(a.astype(str))
print(a.astype(bool))
```
结果如下：
 ```
 [['1.0' '2.0' '3.0' '4.0' '5.0' '2.0']
 ['3.0' '4.0' '5.0' '6.0' '7.0' '8.0']
 ['2.0' '4.0' '5.0' '8.0' '4.0' '4.0']]
[[ True  True  True  True  True  True]
 [ True  True  True  True  True  True]
 [ True  True  True  True  True  True]]
 ```

如果在读取文件时候想要把读取的数据保存在不同的列表中，可以用unpack进行解包，不过返回的数据是转置的行(文件矩阵中每列的数据)，若需要每行数据作为列表内容的话要提前进行转置

### numpy.arrange
生成数组
<font color=red>numpy.arange(start, stop, step, dtype = None)</font>
start —— 开始位置，数字，可选项，默认起始值为0
stop —— 停止位置，数字
step —— 步长，数字，可选项， 默认步长为1，如果指定了step，则还必须给出start。
dtype —— 输出数组的类型。 如果未给出dtype，则从其他输入参数推断数据类型。

### numpy.array()
输入列表可以将列表转化为数组，之后可以用reshape方法把这个数组重新改变成需要的维数

## 类型转换

### 转化数据类类型
array.tolist()等，可以将数组转化为列表等类型

### 将数组转化为一维
chararray.flatten()


## 数组运算
### numpy.vstack
将多个数组堆叠起来
比如
a = numpy.vstack(b,c,d)
即是将b，c，d三个数组的数据合并成一个数组，及向量数增加了，但是每个向量的维数和数据没有变化。（注意这里bcd三个数组的维数必须是一样的）



## numpy.linspace(-3, 3 , 500)

这个例子指的是等距离的生成-3到3的数列，输出数组中有500个元素，返回类型是数组，但是都是1维向量，常用于生成坐标向量。

## numpy.meshgrid

输入坐标向量生成坐标矩阵,搭配numpy.linspace 可以生成一个矩阵面，比如xx, yy = np.meshgrid(np.linspace(-3, 3, 500), np.linspace(-3, 3, 500))，这就相当于生成了x坐标和y坐标都是-3到3的范围，且像素在500*500的，可以搭配matplotlib进行画图，例子可以见learning文件中的sklearn学习的nonlinearSVM例子

## numpy.logical_not();numpy.logical_or();numpy.logical_and();numpy.logical_xor()

用来计算真假值，也就是设置标准将一些数据中的数据转化为True和False

## numpy.random
### numpy.random.randint
numpy.random.randint(最低，最高，size=)

### numpy.random.randn
numpy.random.randn(a, b)生成a个b维数据
生成符合标准正态分布的数或者数组,如果参数量增加就会不断提高数组的嵌套数

### numpy.random.seed
一般搭配randint一起使用，伪随机，可以让randint等随机数生成的方法生成的数据可重复，一般是教学时为了做例子使用。1.22版本numpy已经去掉改成使用SeedSequence.spawn

### numpy.vectorize
向量化函数，将一个函数向量化，使这个函数的的每个变量输入都可以是向量，但是注意输入向量使，要么每个参数的维数都是一样的，要么输入标量。

```python
import numpy as np

def myfunc(a,b):
    if a>b:
        return a-b
    else:
        return a+b

vfunc = np.vectorize(myfunc)

a = vfunc(3,[1,2,3,4])
b = vfunc([1,2,3,4],[1,4,6,7])
```
### Mathematical functions
一些数学中常见的经典函数，比如三角函数、双曲函数、指数、对数等等

### Constants
一些数学常数，比如自然数啊圆周率啊啥的


## CSV

## read data from csv file

使用reader对csv文件进行读取，逐行读取，csv也可以对txt文件进行读取。和numpy读取比起来稍微方便点。
```python
with open('example.txt','r') as csvfile:
    plots = csv reader(csvfile,delemiter = ' ')
    for row in plots:
        x.append(int(row[0]))
        y.append(int(row[1]))
```

# Pytorch
## 创建tensor
### 从list或numpy array获得tensor

x = torch.tensor([[1,-1],[-1,1]])
x = torch.from_numpy([[1,-1],[-1,1]])

### torch.zeros 
x = torch.zeros([2,2])
生成2*2的tensor，数据为0

###  torch.ones
x = torch.ones([1,2,5])
生成元素为1 的2*5的tensor

## tensor编辑
### torch.squeeze()
去掉长度为1的特定维
```python
x =torch.zeros([1,2,3])
x = x.squeeze(0)
```
上述代码先获取了三维的张量，第一维长度为1，可以用squeeze去掉（0是索引）
### torch.unsqueeze()
在tensor的指定索引处获得一个新的维度

### torch.transpose
两个维度对调

###  torch.cat
将张量组合起来（注意只有其他维长度相同时才是可以的）

## 运算
sum(),mean(),pow()
## 更换运行设备（GPU或者CPU）
### to
pytorch 可以选择在gpu上面跑，比如
cpu：x=x.to("cpu")
gpu: x = x.to("cuda")
### torch.cuda.is_available()
检测是否有显卡可以用

###  梯度计算
```python
import torch
import torch
x = torch.tensor([[1.,0.],[-1.,1.]],requires_grad=True)
z = x.pow(2).sum()
z.backward()
print(x.grad)
```
x.grad是tensor的属性，通过backward计算梯度之后才会储存。

## 使用torch建立深度学习模型
### Overview of the DNN training procedure
1. load data: torch.utils.data.Dataset  torch.utils.data.Dataloader
2. define neural work: torch.nn
3. define loss funciton: torch.nn.MSELoss(),torch.nn.CrossEntropyLoss()等
4. optimizer(模型优化器，用于设置模型优化方式): torch.optim.SGD(梯度下降)

### 初始化神经网络(torch.nn.sequential)
nn.sequential()能够生成一个顺序容器，里面包含了多个模型，数据输入第一个模型后，第一个模型的输出会成为下一个模型的输入，一层一层往下传递直到最后一个模型

```python
import torch.nn as nn
class MyMdel(nn.Module):
    def __init__(self):
        super(MyMdel,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(10,32),
            nn.Sigmoid(),
            nn.Linear(32,1)
        )
```
比如上述代码，其数据输入过程如图：
![](vx_images/2631011196714.png)
现有10个数据进入到linear模型，输入32个数据到sigmoid里（激活函数），出来还是32个数据，然后到下一个linearmodel，输出一个数据

### 模型优化器选择(torch.optim)

常用torch.optim.SGD():梯度下降

### 模型训练

example：
```python
#并非可以运行的程序，其中很多参数只是用字母代表并没有设置
impor torch.nn as nn import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader

#初始化模型
class MyModel(nn.Module):
    def __init__(self):
        super(MyMdel,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(10,32),
            nn.Sigmoid(),
            nn.Linear(32,1) 
        )
    def forward(self,x):
        return self.net(x)
    
#训练
dataset = MyDataset(file)                                      #通过MyDataset读取数据
tr_set = DataLoader(dataset,16,shuffle=True)                   #把数据放入DataLoader里
model = MyModel().to(device)                                   #建立初始模型并且移动到某个设备准备优化（cpu或者cuda）
criterion = nn.MSELoss()                                       #设置损失函数
optimizer = torch.optim.SGD(model.parameters(),0.1)            #设置准备使用的优化方法

for epch in range(n_epochs):                                   # 设置迭代次数
    model.train()                                              #把模型设为训练模式
    for x,y in tr_set:
        optimizer.zero_grad()                                  #把优化器的梯度设置为0
        x,y = x.to(device),y.to(device)                        #设置运行设备
        pred = model(x)                                        #预测
        loss = criterion(pred,y)                               #预测值和实际值一起运算损失函数的值
        loss.backward()                                        #计算梯度
        optimizer.step()                                       #更新参数


```
### 模型验证（validation）
```python
model.eval()#把模型设置为验证模式，避免优化使其变化
total_loss = 0#初始化总损失
for x,y in dv_set:#dataloader 中的数据遍历
    x,y = x.to(device),y.to(device)
    with torch.no_grad():#验证过程避免计算梯度，可以提高计算速度
    pred = model(x)
    loss = criterion(pred,y)
    
    total_loss+=loss.cpu().item()*len(x) #计算总loss
    avg_loss = total_loss/len(dv_set.dataset) #计算平均loss
    
```

### 测试集
![](vx_images/5983045199383.png)

### 模型保存和加载（torch.save()、torch.load()）





# Pymatgen

## Structure 类

继承自Istructure类，可以使用Istructure所有方法和属性





