import numpy as np
# import torch.utils.data as data
from sklearn.decomposition import PCA
import random
# import os
import pickle
import h5py
import hdf5storage
from  sklearn import preprocessing
import scipy.io as sio
import os

print(os.getcwd())
# 像素块加边框
def zeroPadding_3D(old_matrix, pad_length, pad_depth = 0):
    new_matrix = np.lib.pad(old_matrix, ((pad_length, pad_length), (pad_length, pad_length), (pad_depth, pad_depth)), 'constant', constant_values=0)
    return new_matrix
# 下标分配(2517*2335,2517,2335,4)
def indexToAssignment(index_, Row, Col, pad_length):
    new_assign = {}

    for counter, value in enumerate(index_):
        assign_0 = value // Col + pad_length
        assign_1 = value % Col + pad_length
        # counter为list中顺序下标，[assign_0, assign_1]为点的每个坐标
        new_assign[counter] = [assign_0, assign_1]
    return new_assign

def assignmentToIndex( assign_0, assign_1, Row, Col):
    new_index = assign_0 * Col + assign_1
    return new_index

# 切割像素点块
def selectNeighboringPatch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row-ex_len,pos_row+ex_len+1), :]
    selected_patch = selected_rows[:, range(pos_col-ex_len, pos_col+ex_len+1)]
    return selected_patch
# 抽样
def sampling(groundTruth):
    labels_loc = {}
    # 找出groundTruth张量中最大的一个值
    m = max(groundTruth)
    for i in range(m):
        # ravel()将张量拉成一维，j为下标 [1,32, 4,5, 7574, 7575]
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)
        # labels_loc[i] 为像素点分类编号 indices为list，包含所有下标 [1,32, 4,5, 7574, 7575]
        labels_loc[i] = indices

    whole_indices = []
    for i in range(m):
        # [      labels_loc[i]中原有的元素 ,      ,     ]中分别为1，2，3，4。。。。等对应的列表，列表中为相对应的下标，whole_indices为顺序列表
        whole_indices += labels_loc[i]

    np.random.shuffle(whole_indices)
    return whole_indices


def load_data_HDF(image_file, label_file):
    image_data = hdf5storage.loadmat(image_file)
    label_data = hdf5storage.loadmat(label_file)
    data_all = image_data['chikusei']  # data_all:ndarray(2517,2335,128)
    label = label_data['GT'][0][0][0]  # label:(2517,2335)

    [nRow, nColumn, nBand] = data_all.shape
    print('chikusei', nRow, nColumn, nBand)
    # np.prod 元素相乘， gt 将长宽相乘作list长度
    gt = label.reshape(np.prod(label.shape[:2]), )
    del image_data
    del label_data
    del label

    data_all = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))  # (111104,204)
    print(data_all.shape)
    # 假设现在构造一个数据集data，将其标准化
    data_scaler = preprocessing.scale(data_all)
    data_scaler = data_scaler.reshape(2517,2335,128)

    return data_scaler, gt

def load_data(image_file, label_file):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)
    data_key = image_file.split('/')[-1].split('.')[0]
    label_key = label_file.split('/')[-1].split('.')[0]
    data_all = image_data[data_key]
    label = label_data[label_key]
    gt = label.reshape(np.prod(label.shape[:2]), )

    data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))  # (111104,204)
    print(data.shape)
    data_scaler = preprocessing.scale(data)
    data_scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1], data_all.shape[2])

    return data_scaler, gt

def getDataAndLabels(trainfn1, trainfn2):
    if ('Chikusei' in trainfn1 and 'Chikusei' in trainfn2):
        # 载入HDF高光谱图像
        Data_Band_Scaler, gt = load_data_HDF(trainfn1, trainfn2)
    else:
        Data_Band_Scaler, gt = load_data(trainfn1, trainfn2)

    del trainfn1, trainfn2
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape

    # SSRN
    # patch_length = 4 # neighbor 9 x 9
    # patch_length = 1 # neighbor 9 x 9
    # patch_length = 2 # neighbor 9 x 9
    patch_length = 3 # neighbor 9 x 9
    # patch_length = 5 # neighbor 9 x 9
    whole_data = Data_Band_Scaler
    # 高宽加4边框
    padded_data = zeroPadding_3D(whole_data, patch_length)
    del Data_Band_Scaler

    np.random.seed(1334)
    # 高宽加4边框
    whole_indices = sampling(gt)
    print('the whole indices', len(whole_indices))  # 520

    nSample = len(whole_indices)
    # [有效编辑样本数, 9, 9, 光谱带长度]
    x = np.zeros((nSample, 2 * patch_length + 1, 2 * patch_length + 1, nBand))
    # [有效编辑样本数] - 1 所有样本标号减 1
    y = gt[whole_indices] - 1  # label 1-19->0-18

    whole_assign = indexToAssignment(whole_indices, whole_data.shape[0], whole_data.shape[1], patch_length)
    print('indexToAssignment is ok')
    for i in range(len(whole_assign)):
        x[i] = selectNeighboringPatch(padded_data, whole_assign[i][0], whole_assign[i][1],
                                      patch_length)
    print('selectNeighboringPatch is ok')

    print(x.shape)
    del whole_assign
    del whole_data
    del padded_data

    imdb = {}
    imdb['data'] = np.zeros([nSample, 2 * patch_length + 1, 2 * patch_length + 1, nBand], dtype=np.float32)  # <class 'tuple'>: (9, 9, 100, 77592)
    imdb['Labels'] = np.zeros([nSample], dtype=np.int64)  # <class 'tuple'>: (77592,)
    # 使用set标签，表示data中前nSample个都是training数据
    imdb['set'] = np.zeros([nSample], dtype=np.int64)

    for iSample in range(nSample):
        imdb['data'][iSample, :, :, :, ] = x[iSample, :, :, :]  # (9, 9, 100, 77592)
        imdb['Labels'][iSample] = y[iSample]  # (77592, )
        if iSample % 100 == 0:
            print('iSample', iSample)

    imdb['set'] = np.ones([nSample]).astype(np.int64)
    print('Data is OK.')

    return imdb

train_data_file = '../datasets/Chikusei/HyperspecVNIR_Chikusei_20140729.mat'
train_label_file = '../datasets/Chikusei/HyperspecVNIR_Chikusei_20140729_Ground_Truth.mat'

imdb = getDataAndLabels(train_data_file, train_label_file)#14类
# 打开Chikusei_imdb_128文件并存储内容
# with open('../datasets/Chikusei_imdb_128_size_3.pickle', 'wb') as handle:
# with open('../datasets/Chikusei_imdb_128_size_5.pickle', 'wb') as handle:
with open('../datasets/Chikusei_imdb_128_size_7.pickle', 'wb') as handle:
# with open('../datasets/Chikusei_imdb_128_size_11.pickle', 'wb') as handle:
    pickle.dump(imdb, handle, protocol=4)

print('Images preprocessed')