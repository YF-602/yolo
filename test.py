from ultralytics import YOLO
from utils.test import *
from utils import *


# 加载预训练模型
model = YOLO('yolo11n.pt')

# 只检测 0: car 1: person 2: bicycle
# COCO 0: person 1: bicycle 2: car
results = model.predict(
    source=['./datasets/images/000016.jpg', './datasets/images/000021.jpg'],
    classes=[0, 1, 2],  # 只检测人、自行车、汽车
    conf=0.50,          # 置信度阈值
    save=True,          # 保存结果图像
    # show=True           # 显示结果

)

boxes = []

for result in results:
    boxes.append(result.boxes)  # Boxes object for bounding box outputs

box = boxes[1]
print(type(box))
# <class 'ultralytics.engine.results.Boxes'>
print(box.xywhn)
# 假设，这张图片检测到3个物体
# tensor([[0.8127, 0.5183, 0.3745, 0.8960],
#         [0.3161, 0.3284, 0.3599, 0.3287],
#         [0.2811, 0.5990, 0.4905, 0.4652]])
print(box.cls)
# tensor([0., 0., 0.])
print(model.names[int(box.cls[0])])
# person
print(box.conf[0])
# tensor(0.8888)