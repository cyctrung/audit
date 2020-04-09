import numpy as np
import os
import cv2
import csv
from ssdpytorch.utils.augmentations import SSDAugmentation  
from glob import glob
import torch

class DataSet():
    def __init__(self,data_root, csv_path, transform = None, phase='train'):
        self.csv_to_data(data_root, csv_path)
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, inputp):
        if type(inputp) == tuple:
            index, raw = inputp
        else:
            index = inputp
            raw = False
        img = cv2.imread(self.imgs[index])
        boxes = self.boxes[index]
        labels = self.labels[index]
        if self.transform:
            img, boxes, labels = self.transform(img, self.boxes[index], self.labels[index])
        img = torch.from_numpy(img).permute(2, 0, 1)
        boxes = torch.FloatTensor(np.vstack(boxes))[:, :]
        labels = torch.from_numpy(labels)[:]
        return img, boxes, labels

    def csv_to_data(self, data_root, csv_path):
        self.boxes = []
        self.imgs = []
        self.labels = []
        with open(csv_path,'r') as f:
            data = csv.reader(f)
            for row in data:
                img_name = row[0]
                x1, y1, x2, y2 = map(float, row[1:5])
                o = 1
                w, h = map(float, row[6:])
                img_path = data_root + '/' + img_name
                if img_path not in self.imgs:
                    if os.path.isfile(img_path):
                        self.imgs.append(img_path)
                        self.boxes.append([])
                        self.labels.append([])
                    else:
                        continue
                scale = np.array([w, h, w, h])
                img_index = self.imgs.index(img_path)
                self.boxes[img_index].append(np.array([x1, y1, x2, y2])/scale)
                self.labels[img_index].append(1)
        self.imgs = np.array(self.imgs)
        self.labels = convert_to_nparray(self.labels)
        self.boxes = convert_to_nparray(self.boxes)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()
        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
        images = torch.stack(images, dim=0)
        return images, boxes, labels

def convert_to_nparray(l):
    
    temp_arr = []
    for arr in l:
        temp_arr.append(np.asarray(arr))
    temp_arr = np.array(temp_arr)
    return temp_arr

if __name__ == "__main__":
    data = DataSet('MiniSKU/train','MiniSKU/annotations/train.csv',SSDAugmentation())
    img, boxes, labels = data[10]
    print(boxes.shape)
    pass
    dist1 = cv2.convertScaleAbs(img)
    w, h, _ = dist1.shape
    for x1, y1, x2, y2 in boxes:
        dist1 = cv2.rectangle(dist1, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), (255, 0, 0), 1)
    cv2.imshow("prview", dist1)
    cv2.waitKey()
    pass


