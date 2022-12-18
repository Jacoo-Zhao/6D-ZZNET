#Extract descriptor for a single image
import torch
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import pdb
import os, time, json
from progress.bar import *
import numpy as np
import models

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
# if torch.cuda.device_count()==1:
torch.cuda.set_device(0 if torch.cuda.device_count()==1 else 2 ) 
print('Current GPU Device: {}'.format(torch.cuda.current_device()))


def extract_single_image(img_path, model):
    """ extract descriptors of one single image
    img_path: string, target image path
    model: model used to extract (netvlad)
    """
    # read image
    img = Image.open(img_path).convert('RGB') # modify the image path according to your need
    pdb.set_trace()
    transformer = transforms.Compose([transforms.Resize((480, 640)), # (height, width)
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.48501960784313836, 0.4579568627450961, 0.4076039215686255],
                                                        std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098])])
    img = transformer(img)

    # use GPU (optional)
    model = model.cuda()
    img = img.cuda()

    # extract descriptor (4096-dim)
    with torch.no_grad():
        des = model(img.unsqueeze(0))[0]
        # des = des.cpu().numpy()
    return des


def get_des_pool(filepath, model):
    """ extract descriptors of one single image
    file_path: file path of the images dataset
    model: model used for feature extraction
    """
    dir = os.listdir(filepath)
    img_num = len(dir)
    dim = 4096
    bar = IncrementalBar('Processing', max=img_num, suffix = '%(percent)d%%')
    des = torch.rand((img_num, dim))
    print("Total images number:{}".format(img_num))
    

    for i,f in enumerate(dir):
        img_path = os.path.join(filepath,f)
        des[i,:]=extract_single_image(img_path, model)
        bar.next()
    bar.finish()
    return des



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


def do_match(des_pool_dir='Data/des_pool_zznet.npy'):
    # load the best model with PCA (trained by our SFRS)
    # model = torch.hub.load('yxgeee/OpenIBL', 'vgg16_netvlad', pretrained=True).eval()
    
    model_pca = model_ibl(pretrained=False) # [B,C,H,W]--->torch.Size([1, 4096])
    model_pca.load_state_dict(torch.load('model_zoo/zznet/model_pca.pt'))
    model_pca.cuda()
    model = model_pca
    # load/process des_pool
    if os.path.exists(des_pool_dir):
        des_pool = torch.tensor(np.load(des_pool_dir)).cuda()
    else:
        # extract des of images    
        filepath = '/cvlabdata2/home/ziyi/6D-Pose/dataset/train/train/rgb'
        des_pool = get_des_pool(filepath, model)
        print(des_pool.shape)
        np.save("des_pool_zznet.npy", des_pool)
        print('-----Train Images Descriptors Pool Finished-----')
 
    # do retrieval
    print("---Start Do Retrieval---")
    q_img_folder = '/cvlabdata2/home/ziyi/6D-Pose/dataset/validation/validation' 
    q_img_dir = os.listdir(q_img_folder) 
    result = {'q_img':[], 
              'retrieval_index':[],
              'similarity':[]}
    result_2 = []
    # Progress Bar 
    with IncrementalBar('Processing', max=len(q_img_dir)) as bar:
        for i in range(len(q_img_dir)):
            q_img = os.path.join(q_img_folder,q_img_dir[i])
            des_query = extract_single_image(img_path=q_img, model=model) # shape:(1,dim_features)
            similarity = torch.cosine_similarity(des_query, des_pool, dim=1)
        
            result['q_img'].append(q_img)
            result['retrieval_index'].append(torch.argmax(similarity).item())
            result['similarity'].append(similarity.max().item())
            result_2.append('q_img:{}, rtv_id:{}, sim:{}'.format(q_img_dir[i],torch.argmax(similarity).item(), similarity.max().item() ))
            # print('Similarity shape:', similarity.shape)
            # print('Max index:',torch.argmax(similarity))
            # print('Max cosine similarity:', similarity.max())
            bar.next()

    # Save result

    # dumps 将数据转换成字符串
    result_json = json.dumps(result,sort_keys=False, indent=4, separators=(',', ': '))
    f = open('result_zznet_format-1.json', 'w')
    f.write(result_json)
    f.close()
    print('Result_format-1 Saved.')

    result_file = 'result_zznet_format-2.json'
    with open(result_file,'w') as f:
        json.dump(result_2,f) 
    print('Result_format-2 saved successfully')


# if __name__ == '__main__':
#     main(des_pool_dir='Data/des_pool_zznet.npy')
    