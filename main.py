import torch
import torch.nn as nn
from torch.autograd import Variable

from dataloader.dataset import ZZNETDataset
from torch.utils.data import DataLoader

import models
from models.netvlad_v1 import NetVLAD, EmbedNet

from loss import HardTripletLoss
from torchvision.models import resnet18

import pdb

torch.cuda.set_device(0 if torch.cuda.device_count()==1 else 1 ) 
print('Current GPU Device: {}'.format(torch.cuda.current_device()))

# 1. Data
dataset_dir = '/cvlabdata2/home/ziyi/6D-Pose/Dataset/train/train'
train_dataset = ZZNETDataset(raw_image=True, grayscale=False, augment=False, root_dir=dataset_dir)
# train_dataset.getitem(0)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=False, num_workers=1)

pdb.set_trace()


# 2. Model
def model_raw():
    """ Inherited from pytorch_NetVlad https://github.com/lyakaap/NetVLAD-pytorch
    """
    # Discard layers at the end of base network
    encoder = resnet18(pretrained=True)
    base_model = nn.Sequential(
        encoder.conv1,
        encoder.bn1,
        encoder.relu,
        encoder.maxpool,
        encoder.layer1,
        encoder.layer2,
        encoder.layer3,
        encoder.layer4,
    )
    dim = list(base_model.parameters())[-1].shape[0]  # last channels (512)
    # Define model for embedding
    net_vlad = NetVLAD(num_clusters=32, dim=dim, alpha=1.0)
    model = EmbedNet(base_model, net_vlad).cuda()
    return model
model = model_raw()

def model_ibl(pretrained=False):
    """ Inherited from openIBL https://github.com/yxgeee/OpenIBL
    """
    base_model = models.create('vgg16', pretrained=True)
    pool_layer = models.create('netvlad', dim=base_model.feature_dim)
    model = models.create('embednetpca', base_model, pool_layer)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url('https://github.com/yxgeee/OpenIBL/releases/download/v0.1.0-beta/vgg16_netvlad.pth', map_location=torch.device('cpu')))
    model.cuda()
    return model()

model_pca = model_ibl(pretrained=False)

# 3. Cirterion and Optimizer
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Define loss
criterion = HardTripletLoss(margin=0.1).cuda()
optimizer = torch.optim.Adagrad(model.parameters(), lr=5e-5, lr_decay=1e-7)

# 4. Train & Test


# This is just toy example. Typically, the number of samples in each classes are 4.
# labels = torch.randint(0, 4, (20, )).long()  ---> by KNN clustering got the labels of the 15000
x = torch.rand(20, 3, 200, 200).cuda()  

# train_loader = DataLoader(dataset=data_train, batch_size=opt.batch, shuffle=True, drop_last=True)
# test_loader = DataLoader(dataset=data_test, batch_size=len(data_test))

# output = model(x)
# triplet_loss = criterion(output, labels)

# pdb.set_trace()
# print(triplet_loss)

triplet_loss = 100
for epoc in range(500):
    if triplet_loss>=1e-3:
        pdb.set_trace()
        y_pred = model(x)
        triplet_loss = criterion(y_pred, labels)
        if epoc%5==0:
            print(epoc, triplet_loss)
        optimizer.zero_grad()
        triplet_loss.backward()
        optimizer.step()
print(epoc, triplet_loss)

# Test model
