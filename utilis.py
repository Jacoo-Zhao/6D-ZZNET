# -*- coding: utf-8 -*-

import os
import argparse
import torch
import numpy as np
import pdb
import pickle
from skimage import io
from skimage import color
from skimage.transform import rotate, resize
from PIL import Image
from torchvision import transforms
import torchvision
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from random import sample
from sklearn.manifold import Isomap
from sklearn.datasets import load_digits
from progress.bar import *


def sampling_location_visu(pose_path='Data/poses.csv'):
    poses = np.loadtxt(pose_path, delimiter=',')

    poses = poses[poses[:,3].argsort()]
    # Creating dataset
    # lat = poses[:,1][0:500]
    # log = poses[:,2][0:500]
    # hei = poses[:,3][0:500]
    
    lat = poses[:,1]
    log = poses[:,2]
    hei = poses[:,3]
    # # data_sample = 200
    # lat_mini = sample(lat, data_sample)
    # log_mini = sample(log, data_sample)
    # hei_mini = sample(hei, data_sample)

    "https://www.geeksforgeeks.org/3d-scatter-plotting-in-python-using-matplotlib/"
    # Creating figure
    fig = plt.figure(figsize=(40, 28))
    ax = plt.axes(projection="3d")
    
    # Creating plot
    ax.scatter3D(log, lat, hei, linewidths=0.01, color="green")
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.title("simple 3D scatter plot")
    plt.savefig('demo.png')

    # fig = plt.figure(figsize=(40, 28))
    # ax1 = plt.axes(projection="3d")
    # ax1.scatter3D(log, lat, hei,  color="green")  #绘制散点图
    # ax1.plot3D(log, lat, hei,'gray')    #绘制空间曲线
    # plt.savefig('demo2.png')

    # fig = plt.figure(figsize=(80, 56))
    # ax = plt.axes(projection="3d")
    # ax.plot(log, lat, hei, label='parametric curve')
    # ax.legend()
    # plt.savefig('demo3.png')

    # new a figure and set it into 3d
    fig = plt.figure(figsize=(80, 56))
    ax = plt.axes(projection="3d")
    # set figure information
    ax.set_title("3D_Curve")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    # draw the figure, the color is r = read
    ax.plot(log[0:50], lat[0:50], hei[0:50], c='r')
    plt.savefig('demo3.png')
    pdb.set_trace()
    return 0

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


def euclidean_dist_cuda(x, y):
    """
    Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
    Returns:
        dist: pytorch Variable, with shape [m, n]
    """    
    # 判断x是否为tensor类型
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    if not torch.is_tensor(y):
        y = torch.tensor(y)
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
    return dist
 
def tuple_formulate(pose_path, data_tuple_path):
    """
    Formulate a single tuple of a query img

    Args:
    dir: string, source dir of the trainingset,
    data_tuple_path: alradey written tuple file
    
    Returns: 
    tuple_item: a tuple structure comprised of 3 parts, query image -- 32 positive images\\
         -- 32 negative images ((15000,1),(15000,32),(15000,32))=(q_img, pos, neg)
    """

    base = 'Data'
    pose_path_new = []
    pose_path = ['poses.csv', 'poses_synth-real-matching.csv', 'poses_synth-real-matching.csv']
    for  i in pose_path:
        pose_path_new.append(os.path.join(base, i))

    # read/write file
    try:
        with open(data_tuple_path, "rb") as f:
            data = pickle.load(f)
        print('File {} already exists.'.format(data_tuple_path))
    except FileNotFoundError:
        print('File not found, start processing.')
        # pp = {'p1':[], 'p2':[], 'p3':[]}
        # for i in poses_path_new:
            # pp[i] = torch.from_numpy(np.loadtxt(i, delimiter=',')) #(15000,7) including index.
        pp0 = torch.from_numpy(np.loadtxt(pose_path_new[0], delimiter=','))
        pp1 = torch.from_numpy(np.loadtxt(pose_path_new[1], delimiter=','))
        pp2 = torch.from_numpy(np.loadtxt(pose_path_new[2], delimiter=','))

        poses = torch.cat((pp0, pp1, pp2),dim=0)
        # pdb.set_trace()

        poses_loc = poses[:,1:4] # n*3
        poses_ang = poses[:,4:7] # n*3
        poses_loc_ang = poses[:,1:]

        dist_loc = euclidean_dist_cuda(poses_loc, poses_loc)
        dis_ang = euclidean_dist_cuda(poses_ang, poses_ang)

        dis = dist_loc + 2e0*dis_ang

        # sim_ang = torch.zeros(poses_ang.shape[0], poses_ang.shape[0])
        # with IncrementalBar('Processing', max=poses_ang.shape[0]) as bar:
        #     for i in range(poses_ang.shape[0]):
        #         for j in range(poses_ang.shape[0]):
        #             sim_ang[i,j]=torch.cosine_similarity(poses_ang[i].unsqueeze(0),poses_ang[j].unsqueeze(0))  
        #         bar.next()
        #     bar.finish()

        # _, idx_sort_loc = torch.sort(dist_loc, dim=1, descending=True)
        # _, idx_sort_ang = torch.sort(sim_ang, dim=1, descending=True)
        _, idx_sort = torch.sort(dis, dim=1, descending=True)

        # pdb.set_trace()
        q_img = torch.arange(0, 15000+1197+1197).reshape((15000+1197+1197, 1))
        pos_pool = idx_sort[:, -33:-1]
        neg_pool = idx_sort[:,0:32]
        x = torch.cat((q_img, pos_pool, neg_pool), dim=1)
        data = (x[:,0],x[:,1:33],x[:,33:65])

        with open(data_tuple_path, 'wb') as f:
            pickle.dump({'data': data}, f)
        print('Tuple file saved in: {}'.format(data_tuple_path))
    return data

def save_tuple_wgs(index, pos_img_id, neg_img_id, pose_path='Data/poses.csv', save_path='Data/Tuple_loc+ang.csv'):
    """save (Latitude, Logitude, height)
    Args:
        index: int. index of query image
        pos_img_id: int. index of positive image corresponding to query image
        neg_img_id: int. index of negative image corresponding to quey image
        pose_path: str. wgs data of all rgb images.

    Function:
        write wgs position to 'Data/Tuple_Log.csv'
    """ 
    columns=['q_id', 'p_id', 'n_id', 'q_lat','q_log','q_heg','p_lat','p_log', 'p_heg', 'neg_lat', 'neg_log', 'neg_heg']
    ids= [index, pos_img_id, neg_img_id]
    try:
        df = pd.read_csv(save_path, header=0) # 第一行  
    except FileNotFoundError:
        df = pd.DataFrame(columns=columns)
    # save (Latitude, Logitude, height)
    index = index
    pos_img_id = pos_img_id.item()
    neg_img_id = neg_img_id.item()
    poses = torch.from_numpy(np.loadtxt(pose_path, delimiter=',')) #(15000,7)
    q=poses[index][1:4].numpy().tolist()
    p = poses[pos_img_id][1:4].numpy().tolist()
    n = poses[neg_img_id][1:4].numpy().tolist() 
    data = [ids+q+p+n] 
    q_p_n_wgs = pd.DataFrame(data=data, columns=columns)
    df=pd.concat([df, q_p_n_wgs], ignore_index=True)
    DataFrame(df).to_csv(save_path, index=False, header=True)

def geodesic_dis(x=np.random.randint(1, 10, (2,3))):
    # X, _ = load_digits(return_X_y=True)
    # data = np.load("samples_data.npy")
    arr = np.add(np.ones((10,8)), np.arange(8)).astype(int).T
    embedding  = Isomap(n_components=2,n_neighbors=5,path_method="auto")
    data_2d = embedding .fit_transform(arr)
    geo_distance_matrix = embedding.dist_matrix_ # 测地距离矩阵，shape=[n_sample,n_sample]
    pdb.set_trace()

# def rotation_sim(deta):
#     """ 
#     Args:
#         x: m*d,
#         y: n*d,
#     Return: 
#         y: float, m*n,
#     """

#         if deta<=180:
#             y = deta/180
#         else:
#             y = (dets-180)/180
        
#     return y

if __name__=='__main__':
    torch.cuda.set_device(0 if torch.cuda.device_count()==1 else 3 ) 
    parser = argparse.ArgumentParser()
    parser.add_argument('--echo',default="Processing...", type=str, help ='echo')
    args = parser.parse_args()
    print(args.echo)

    data_tuple_path = 'Data/Tuple_loc_ang(L2Norm).pickle'
    tuple_formulate(pose_path='poses.csv', data_tuple_path=data_tuple_path,)
   
    # read_img('/cvlabdata2/home/ziyi/6D-Pose/Dataset/train/images/Echendens-LHS_00000.png')
    
    # for i in range(10):
        # save_tuple_wgs(index=0, pos_img_id=5701, neg_img_id=10450, posepath='Data/poses.csv')

    # sampling_location_visu()
    # arr = np.add(np.ones((10,8)), np.arange(8)).astype(int).T

    print("Succeed!")
