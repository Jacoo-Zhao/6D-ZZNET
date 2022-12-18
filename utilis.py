# -*- coding: utf-8 -*-

import os
import argparse
import torch
import numpy as np
import pdb
import numpy
import pickle
from skimage import io
from skimage import color
from skimage.transform import rotate, resize
from PIL import Image
from torchvision import transforms
import torchvision
import pandas as pd
from pandas import DataFrame

def read_files():
    filepath = '/cvlabdata2/home/ziyi/6D-Pose/dataset/train_mini_mini/images'
    dir = os.listdir(filepath)
    for i, f in enumerate(dir):
        img_path = os.path.join(filepath,f)
    print(i, img_path)

    for root, dirs, files in os.walk(file):
        print(os.walk(file)[0])
    for file in files:
        file_num = len(files)
        path = os.path.join(root, file)
        print(file_num)

def read_img(img_path):
    img1 = io.imread(img_path)
    img2 = np.array(Image.open(img_path).convert('RGB')) # ==>(H,W,channel)
    pdb.set_trace()
    image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(480),
            transforms.ToTensor()
        ])
    pdb.set_trace()
    img2_tf = image_transform(img2)
    # new_qImg = Image.fromarray(img2_tf.numpy())
    # new_qImg.save('Data/demo-qImg.png')
    torchvision.utils.save_image(img2_tf, 'Data/demo-qImg-SaveFromTensor.jpg')



def pickle_reader(filepath):
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    pdb.set_trace()

def EuclideanDistances(A, B):
    BT = B.transpose()
    vecProd = A * BT
    SqA =  A.getA()**2
    sumSqA = numpy.matrix(numpy.sum(SqA, axis=1))
    sumSqAEx = numpy.tile(sumSqA.transpose(), (1, vecProd.shape[1]))    
    SqB = B.getA()**2
    sumSqB = numpy.sum(SqB, axis=1)
    sumSqBEx = numpy.tile(sumSqB, (vecProd.shape[0], 1))    
    SqED = sumSqBEx + sumSqAEx - 2*vecProd   
    ED = (SqED.getA())**0.5
    return numpy.matrix(ED)

def euclidean_dist_cuda(x, y):
    """
    Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
    Returns:
        dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    # xx经过pow()方法对每单个数据进行二次方操作后，在axis=1 方向（横向，就是第一列向最后一列的方向）加和，此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    # yy会在最后进行转置的操作
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    # torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None)，这行表示的意思是dist - 2 * x * yT 
    # dist.addmm_(1, -2, x, y.t())
    dist = dist - 2*(torch.mm(x.float(),y.t().float()))
    # dist = torch.addmm(dist, x, y.t(), beta=1, alpha=-2)
    # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    # print(dist.shape)
    return dist
 
def tuple_formation(img_path, pose_path, data_tuple_path, dataset_dir):
    """
    Formulate a single tuple of a query img

    Args:
    dir: string, source dir of the trainingset,
    data_tuple_path: alradey written tuple file
    
    Returns: 
    tuple_item: a tuple structure comprised of 3 parts, query image -- 32 positive images\\
         -- 32 negative images ((15000,1),(15000,32),(15000,32))=(q_img, pos, neg)
    """

    base = '/cvlabdata2/home/ziyi/6D-Pose/dataset/train/train'
    # dataset_dir = os.path.join(base, dataset_dir)
    # img_path = os.path.join(base, img_path)
    pose_path = os.path.join(base, pose_path)
    # read/write file
    try:
        with open(data_tuple_path, "rb") as f:
            data = pickle.load(f)
        print('File {} already exists.'.format(data_tuple_path))
    except FileNotFoundError:
        print('Fild not found, start processing.')
        poses = torch.from_numpy(np.loadtxt(pose_path, delimiter=',')) #(15000,7) including index.
        poses_tsl = poses[:,1:4] 
        dist = euclidean_dist_cuda(poses_tsl, poses_tsl)
        _, idx_sort = torch.sort(dist, dim=1, descending=True)
        q_img = torch.arange(0, 15000).reshape((15000, 1))
        pos_pool = idx_sort[:, -33:-1]
        neg_pool = idx_sort[:,0:32]
        x = torch.cat((q_img, pos_pool, neg_pool), dim=1)
        pdb.set_trace()
        data = (x[:,0],x[:,1:33],x[:,33:65])

        with open(data_tuple_path, 'wb') as f:
            pickle.dump({'data': data}, f)
    return data

def save_tuple_wgs(index, pos_img_id, neg_img_id, pose_path='Data/poses.csv', save_path='Data/Tuple_WGS.csv'):
    """save (Latitude, Logitude, height)
    Args:
        index: int. index of query image
        pos_img_id: int. index of positive image corresponding to query image
        neg_img_id: int. index of negative image corresponding to quey image
        pose_path: str. wgs data of all rgb images.

    Function:
        write wgs position to 'Data/Tuple_Log.csv'
    """ 
    columns=['q_lat','q_log','q_heg','p_lat','p_log', 'p_heg', 'neg_lat', 'neg_log', 'neg_heg']
    try:
        df = pd.read_csv(save_path, header=0) # 第一行  
    except FileNotFoundError:
        df = pd.DataFrame(columns=columns)
    # save (Latitude, Logitude, height)
    index = index
    pos_img_id = pos_img_id
    neg_img_id = neg_img_id
    poses = torch.from_numpy(np.loadtxt(pose_path, delimiter=',')) #(15000,7)
    q=poses[index][1:4].numpy().tolist()
    p = poses[pos_img_id][1:4].numpy().tolist()
    n = poses[neg_img_id][1:4].numpy().tolist() 
    data = [q+p+n] 
    q_p_n_wgs = pd.DataFrame(data=data, columns=columns)
    df=pd.concat([df, q_p_n_wgs], ignore_index=True)
    DataFrame(df).to_csv(save_path, index=False, header=True)


if __name__=='__main__':
    torch.cuda.set_device(0 if torch.cuda.device_count()==1 else 3 ) 
    parser = argparse.ArgumentParser()
    parser.add_argument('--echo',default="Processing!", type=str, help ='echo')
    args = parser.parse_args()
    print(args.echo)

    # eg_a = torch.randn(15000,4096).cuda()
    # eg_b = torch.randn(15000,4096).cuda()
    # dataset_tuple_initialize('des_pool.npy')

    # pdb.set_trace()
    data_tuple_path = 'Data/data_tuple.pickle'
    pickle_reader(data_tuple_path)
    # tuple_formation(img_path='', pose_path='poses/poses.csv', data_tuple_path=data_tuple_path, dataset_dir='', )
    # read_img('/cvlabdata2/home/ziyi/6D-Pose/Dataset/train/images/Echendens-LHS_00000.png')
    # for i in range(10):
        # save_tuple_wgs(index=0, pos_img_id=5701, neg_img_id=10450, posepath='Data/poses.csv')

    print("Succeed.")
