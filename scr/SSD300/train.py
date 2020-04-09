from tqdm import tqdm
import time
import torch
import numpy as np
from dataset import DataSet
from loss2 import MultiBoxLoss
from ssd300 import SSD
from ssdpytorch.utils.augmentations import SSDAugmentation
from torch.utils.data import DataLoader
from torch.autograd import Variable


EPOCH = 50
print_feq = 2

def collate_fn(batch):
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

train_dataset = DataSet('SKU110K/images','MiniSKU/annotations/train.csv',SSDAugmentation())
train_loader = DataLoader(train_dataset, 2, num_workers=2, collate_fn=collate_fn)
val_dataset = DataSet('SKU110K/images','MiniSKU/annotations/val.csv',SSDAugmentation())
val_loader = DataLoader(val_dataset, 2, num_workers=2, collate_fn=collate_fn)
model = SSD().cuda()
optimizer = torch.optim.SGD(model.parameters(),lr=0.0001, momentum=0.9)
criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy)
model.to('cuda')

best_loss = 10000
best_state = None

for epoch in range(1, EPOCH+1):
    model.train()
    train_loss = []
    for step, (img, boxes, labels) in enumerate(train_loader):
        time_1 = time.time()
        img = img.cuda()
#         box = torch.cat(box)
        boxes = [box.cuda() for box in boxes]
#         label = torch.cat(label)
        labels = [label.cuda() for label in labels]
        
        pred_loc, pred_sco = model(img)
        
        loss = criterion(pred_loc, pred_sco, boxes, labels)
        
         # Backward prop.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
#         losses.update(loss.item(), images.size(0))
        train_loss.append(loss.item())
        if step % print_feq == 0:
            print('epoch:', epoch, 
                  '\tstep:', step+1, '/', len(train_loader) + 1,
                  '\ttrain loss:', '{:.4f}'.format(loss.item()),
                  '\ttime:', '{:.4f}'.format((time.time()-time_1)*print_feq), 's')
    
    model.eval()
    valid_loss = []
    for step, (img, boxes, labels) in enumerate(tqdm(val_loader)):
        img = img.cuda()
        boxes = [box.cuda() for box in boxes]
        labels = [label.cuda() for label in labels]
        pred_loc, pred_sco = model(img)
        loss = criterion(pred_loc, pred_sco, boxes, labels)
        valid_loss.append(loss.item())
        
    print('epoch:', epoch, '/', EPOCH,
            '\ttrain loss:', '{:.4f}'.format(np.mean(train_loss)),
            '\tvalid loss:', '{:.4f}'.format(np.mean(valid_loss)))
    if np.mean(valid_loss) < best_loss:
        best_loss = np.mean(valid_loss)
        best_state = model.state_dict()
torch.save(best_state, 'ssd.pth')
