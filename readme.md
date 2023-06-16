# 接口调用实例说明

### `__init__()`

```python
def __init__(self)
```

初始化`AiGcMn`类的对象，加载经过训练的生成器模型的参数。

### `generate()`

```python
def generate(self, labels)
```

#### 参数

- `labels`（必需）：一个大小为`n` x 1的张量，其中`n`是要生成的图像数量。`labels`张量包含0到9之间的整数，用于指定要生成的数字类别。

#### 返回值

- 一个大小为`n` x 1 x 28 x 28的张量，表示生成的手写数字图像。其中，`n`是要生成的图像数量，1 x 28 x 28是每个图像的大小。

### 示例代码

```python
import torch
from aigcnm import AiGcMn
import matplotlib.pyplot as plt

A = AiGcMn()		#A为接口对象

test_tensor = torch.tensor([8, 2, 3, 6, 7, 8, 7, 9 ,1])	
gen_tensor = A.generate(test_tensor)	#gen_tensor保存了一个大小为8*1*28*28的张量

length = gen_tensor.size()[0]
gen_imgs = gen_tensor.view(-1, 28, 28).numpy()
gen_imgs = 0.5 * gen_imgs + 0.5

fig, axs = plt.subplots(1,
                        length,
                        figsize=(length, 1),
                        sharey=True,
                        sharex=True)

for i in range(length):
    axs[i].imshow(gen_imgs[i], cmap='gray')
    axs[i].axis('off')

fig.show()
```

上述示例代码接收一个包含`8, 2, 3, 6, 7, 8, 7, 9 ,1`的张量，并生成一个横排的图像。