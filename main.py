import torch
import torch.nn as nn
from torch.autograd import Variable

from dataloader.dataset import ZZNETDataset
from torch.utils.data import DataLoader

import models
from models.netvlad_v1 import NetVLAD, EmbedNet

from loss import HardTripletLoss, AccumLoss, zznet_triplet
from torchvision.models import resnet18
from tqdm import tqdm
from des_extr import match

import pdb
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

torch.cuda.set_device(0 if torch.cuda.device_count()==1 else 1 ) 

print('Current GPU Device: {}'.format(torch.cuda.current_device()))
is_cuda = torch.cuda.is_available()

# 1. Data
batch_size = 32
dataset_dir = '/cvlabdata2/home/ziyi/6D-Pose/Dataset/train/train'
train_dataset = ZZNETDataset(raw_image=False, grayscale=False, augment=True, root_dir=dataset_dir)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

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

def model_ibl(pretrained=False):
    """ Inherited from openIBL https://github.com/yxgeee/OpenIBL
    """
    base_model = models.create('vgg16', pretrained=True)
    pool_layer = models.create('netvlad', dim=base_model.feature_dim)
    model = models.create('embednetpca', base_model, pool_layer)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url('https://github.com/yxgeee/OpenIBL/releases/download/v0.1.0-beta/vgg16_netvlad.pth', map_location=torch.device('cpu')))
    model.cuda()
    return model

model = model_raw() # [B,C,H,W]--->torch.Size([1, 16384])
model_pca = model_ibl(pretrained=False) # [B,C,H,W]--->torch.Size([1, 4096])
# pdb.set_trace()

# 3. Cirterion and Optimizer
lr = 1e-1
criterion = zznet_triplet(margin=0.1).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 4. Train & Test
epoch = 50
model_pca.train()

pdb.set_trace()
with tqdm(range(epoch), desc=f'Training model', unit="epoch") as tepoch:
# with tqdm(range(epoch),  desc=f'Training model',  unit="epoch") as tepoch:
    for epoch in tepoch:
        # print("Epoch: {}".format(epoch))
        tr_loss = AccumLoss()
        # pdb.set_trace()
        for i, (q_img, p_img, n_img, _, _, _,) in enumerate(train_loader):
            # print("Epoch: {}, Index:{}".format(epoch, i))
            if is_cuda:
                q_img = q_img.cuda()
                p_img = p_img.cuda()
                n_img = n_img.cuda()
            else:
                q_img = q_img
                p_img = p_img
                n_img = n_img

            des_q = model_pca(q_img)
            des_p = model_pca(p_img)
            des_n = model_pca(n_img)
            
            margin = 5e-1
            loss = criterion(des_q, des_p, des_n, margin)
            optimizer.zero_grad()
            # loss.backward()
            loss.backward(loss.clone().detach())
            tr_loss.update(loss.cpu().data.numpy() * batch_size, batch_size)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
        tr_loss = tr_loss.avg
        tepoch.set_postfix(train_loss=tr_loss.item())

# 5. Save model--->funciton
model_dir = 'model_zoo/zznet/model_pca.pt'
model._save_to_state_dict()
torch.save(model_pca.state_dict(), model_dir)

# 5. Retrieval
do_match(des_pool_dir='Data/des_pool_zznet.npy')


# 6. 2d-2d match : SuperGlue
