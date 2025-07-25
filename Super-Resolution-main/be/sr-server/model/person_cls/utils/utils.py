import numpy as np
from PIL import Image


# 对输入图像进行resize
def letterbox_image(image, size):
    """
    resize image with unchanged aspect ratio using padding
    """
    iw, ih = image.size
    h, w = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image


# 获得类
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


# 预处理训练图片
def preprocess_input(x):
    x /= 127.5
    x -= 1.
    return x


# 将图像转换成RGB图像，防止灰度图在预测时报错。
# 代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

