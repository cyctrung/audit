from tqdm import tqdm
import cv2
import time
import torch
import numpy as np
from dataset import DataSet
from loss2 import MultiBoxLoss
from ssd300 import SSD
from ssdpytorch.utils.augmentations import SSDAugmentation
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib as plt
from PIL import ImageDraw


model = SSD()
model.cuda()
model.load_state_dict(torch.load('ssd.pth'))
test_dataset = DataSet('MiniSKU/test','MiniSKU/annotations/test.csv', SSDAugmentation(scale_only=True))
test_loader = DataLoader(test_dataset, 1, num_workers=2, collate_fn=test_dataset.collate_fn)

def test(n=5):
    d = []
    for i, (img, boxes, labels) in enumerate(test_loader):
        predicted_locs, predicted_scores = model(img[0].unsqueeze(0).cuda())
        det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=0.2,
                                                                 max_overlap=0.5, top_k=200)
        det_boxes = det_boxes[0].to('cpu')
        img = img[0].permute(1,2,0)
        img = torch.squeeze(img)
        print(img.shape)
        dist1 = cv2.convertScaleAbs(img.numpy())
        w, h, _ = dist1.shape
        origin_dims = torch.FloatTensor([w,h,w,h]).unsqueeze(0)
        det_boxes = det_boxes * origin_dims
        
        boxes = det_boxes
        for x1, y1, x2, y2 in boxes:
            dist1 = cv2.rectangle(dist1, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
        cv2.imshow("prview", dist1)
        cv2.waitKey()
        cv2.imwrite('test/'+str(i)+'.jpg', dist1)
        d.append(dist1)
    return d

test()