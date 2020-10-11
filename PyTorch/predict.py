import pycocotools
import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image, ImageDraw
import pandas as pd
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch, evaluate
import utils
import transforms as T
from torchvision.transforms import ToTensor as tens
import cv2 as cv
import math


os.chdir("C:/torch_classifier")


def parse_one_annot(path_to_data_file, filename):
    data = pd.read_csv(path_to_data_file)
    boxes_array = data[data["filename"] == filename][["xmin", "ymin",
                                                      "xmax", "ymax"]].values
    return boxes_array


class RaccoonDataset(torch.utils.data.Dataset):
    def __init__(self, root, data_file, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = sorted(os.listdir(os.path.join(root, "images")))
        self.path_to_data_file = data_file

    def __getitem__(self, idx):
        # load images and bounding boxes
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        box_list = parse_one_annot(self.path_to_data_file, self.imgs[idx])
        boxes = torch.as_tensor(box_list, dtype=torch.float32)
        num_objs = len(box_list)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        if self.transforms is not None:
            img, target = self.transforms(img, target)
            return img, target

    def __len__(self):
        return len(self.imgs)


def get_transform(train):
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def get_model(num_classes):
    # load an object detection model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


device = torch.device("cuda")
loaded_model = get_model(num_classes=2)
loaded_model.load_state_dict(torch.load('building_detection.pth'))
loaded_model.to(device)
loaded_model.eval()

dataset = RaccoonDataset(root="", data_file="label.csv", transforms=get_transform(train=True))


def predict(img, model, filename):
    img_tens = tens()(img)
    cuda_tens = img_tens.cuda()
    cuda_tens.to(device)
    image = Image.fromarray(cuda_tens.mul(255).permute(1, 2, 0).byte().cpu().numpy())
    draw = ImageDraw.Draw(image)
    box_count=0
    with torch.no_grad():
        prediction = loaded_model([cuda_tens])
    for element in range(len(prediction[0]["boxes"])):
        boxes = prediction[0]["boxes"][element].cpu().numpy()
        score = np.round(prediction[0]["scores"][element].cpu().numpy(), decimals=4)
        if score > 0.5:
            draw.rectangle([(boxes[0], boxes[1]), (boxes[2], boxes[3])], outline="red", width=3)
            draw.text((boxes[0], boxes[1]), text=str(score))
            box_count +=1
    save1 = np.asarray(image)
    cv.imwrite("./save/"+str(filename)+".png", save1)
    return box_count


a = cv.imread("santa-rosa-wildfire_00000038_pre_disaster.png")
b = cv.imread("santa-rosa-wildfire_00000038_post_disaster.png")


def disaster_compare(image_before, image_after, model):
    c1 = predict(image_before, model=model, filename="before")
    c2 = predict(image_after, model=model, filename="after")
    print(c1)
    print(c2)
    try:
        res = c2/c1
    except:
        res = "inf"
    with open("res.txt", "w") as f:
        f.write("relative damage is " + str(res))

disaster_compare(a, b, loaded_model)