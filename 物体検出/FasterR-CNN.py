import cv2
import torch
import torchvision

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image #, ImageDraw, ImageFont
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

image1_file_path = './car.jpg'
image2_file_path = './horse.jpg'

img1 = Image.open(image1_file_path).convert("RGB")
img2 = Image.open(image2_file_path).convert("RGB")

def get_object_detection_model():
    # load an object detection model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True) 
    return model

# get the model using our helper function
model = get_object_detection_model()

# move model to the right device
model.to(device)

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

img1 = transform(img1)
img2 = transform(img2)

score_threshold = 0.9

model.eval()
with torch.no_grad():
    predictions = model([
        img1.to(device), img2.to(device)
    ])

    # filter out predictions with low scores
    _predictions = []
    for pred in predictions:
        mask = np.argmax(pred['scores'])
        _predictions.append(
            {
                'boxes': pred['boxes'][mask],
                'labels': pred['labels'][mask],
                'scores': pred['scores'][mask],
            }
        )
    predictions = _predictions

print(predictions)

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

print(COCO_INSTANCE_CATEGORY_NAMES[predictions[0]['labels'].item()], COCO_INSTANCE_CATEGORY_NAMES[predictions[1]['labels'].item()])

font = cv2.FONT_HERSHEY_SIMPLEX 
font_scale = 2
color = (255, 0, 0)
thickness = 3
delta = 20

inputs = [cv2.imread(path) for path in [image1_file_path, image2_file_path]]
for i,img in enumerate(inputs):

    _box, _label, _score = predictions[i]['boxes'], predictions[i]['labels'], predictions[i]['scores']
    # get prediction results: bounding box, label(_name), prediction score

    box = {k:int(v) for (k,v) in zip(['x0', 'y0', 'x1', 'y1'], _box.tolist())}
    # If you have a one element tensor, use .item() to get the value as a Python number
    label = _label.item()
    label_name = COCO_INSTANCE_CATEGORY_NAMES[label]
    score = _score.item()

    # draw bounding box, label and prediction score
    img = cv2.rectangle(img, (box['x0'], box['y0']), (box['x1'], box['y1']), 
                            color, thickness)
    img = cv2.putText(img, '{} {}'.format(label_name, '{:.1%}'.format(score)), (box['x0'], box['y0']-delta), 
                          font, font_scale, color, thickness, cv2.LINE_AA)

    # display
    plt.figure(figsize = (12,9))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()