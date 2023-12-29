import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
import scipy as sp
import scipy.stats
import random
import scipy.io as sio
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.decomposition import FactorAnalysis
from torchvision import transforms
from skimage.transform import resize
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m,h

from operator import truediv
def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc



import torch.utils.data as data


class matcifar(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(self, imdb, train, d, medicinal):

        self.train = train  # training set or test set
        self.imdb = imdb
        self.d = d
        self.x1 = np.argwhere(self.imdb['set'] == 1) #返回数组中值为1的索引
        self.x2 = np.argwhere(self.imdb['set'] == 3)
        self.x1 = self.x1.flatten()
        self.x2 = self.x2.flatten()
        #        if medicinal==4 and d==2:
        #            self.train_data=self.imdb['data'][self.x1,:]
        #            self.train_labels=self.imdb['Labels'][self.x1]
        #            self.test_data=self.imdb['data'][self.x2,:]
        #            self.test_labels=self.imdb['Labels'][self.x2]

        if medicinal == 1:
            self.train_data = self.imdb['data'][self.x1, :, :, :]
            self.train_labels = self.imdb['Labels'][self.x1]
            self.test_data = self.imdb['data'][self.x2, :, :, :]
            self.test_labels = self.imdb['Labels'][self.x2]

        else:
            self.train_data = self.imdb['data'][:, :, :, self.x1]
            self.train_labels = self.imdb['Labels'][self.x1]
            self.test_data = self.imdb['data'][:, :, :, self.x2]
            self.test_labels = self.imdb['Labels'][self.x2]
            if self.d == 3:
                self.train_data = self.train_data.transpose((3, 2, 0, 1))  ###(9, 9, 100, 77592) -> #(77592, 100, 9, 9)
                self.test_data = self.test_data.transpose((3, 2, 0, 1))
            else:
                # self.train_data = self.train_data.transpose((3, 0, 2, 1))
                # self.test_data = self.test_data.transpose((3, 0, 2, 1))
                self.train_data = self.train_data.transpose((3, 2, 0, 1))
                self.test_data = self.test_data.transpose((3, 2, 0, 1))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:

            img, target = self.train_data[index], self.train_labels[index]
        else:

            img, target = self.test_data[index], self.test_labels[index]

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


# def sanity_check(all_set):
#     nclass = 0
#     nsamples = 0
#     all_good = {}
#     #设置混乱程度
#     noise_percentage = 1
#     correct_num = int(200 * (1 - noise_percentage))
#     noise_num = 200 - correct_num
#     for class_ in all_set:
#         if len(all_set[class_]) >= 200:
#             all_good[class_] = all_set[class_][:correct_num]
#             for kk in range(noise_num):
#                 noise_label = class_
#                 while(noise_label == class_):
#                     noise_label = random.randint(0, 15)  # 生成1到10之间的随机整数
#                 noise_sample = random.randint(correct_num,220)
#                 print(f"class is {class_},noise is {noise_label},noise_sample is  {noise_sample}")
#                 all_good[class_].append(all_set[noise_label][noise_sample])
#             nclass += 1
#             nsamples += len(all_good[class_])
#     print('the number of class:', nclass)
#     print('the number of sample:', nsamples)
#     return all_good

def sanity_check(all_set):
    nclass = 0
    nsamples = 0
    all_good = {}
    for class_ in all_set:
        if len(all_set[class_]) >= 200:
            all_good[class_] = all_set[class_][:200]
            nclass += 1
            nsamples += len(all_good[class_])
    print('the number of class:', nclass)
    print('the number of sample:', nsamples)
    return all_good

def flip(data):
    # 生成原始数据一样维度的全0张量
    y_4 = np.zeros_like(data) #生成一个和data一样的0矩阵
    y_1 = y_4
    y_2 = y_4
    first = np.concatenate((y_1, y_2, y_1), axis=1)
    second = np.concatenate((y_4, data, y_4), axis=1)
    third = first
    Data = np.concatenate((first, second, third), axis=0)
    return Data

def load_data(image_file, label_file):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)

    data_key = image_file.split('/')[-1].split('.')[0]
    label_key = label_file.split('/')[-1].split('.')[0]
    data_all = image_data[data_key]  # dic-> narray , KSC:ndarray(512,217,204)
    GroundTruth = label_data[label_key]
    #IP 145*145*200：145*145
    [nRow, nColumn, nBand] = data_all.shape
    print(data_key, nRow, nColumn, nBand)
    # 数据中前两个维度相乘
    data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))  # (111104,204)
    # 数据标准化
    data_scaler = preprocessing.scale(data)  # (X-X_mean)/X_std,
    Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1],data_all.shape[2])

    return Data_Band_Scaler, GroundTruth  # image:(512,217,207),label:(512,217)

def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1/25):
    alpha = np.random.uniform(*alpha_range)
    noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
    return alpha * data + beta * noise

def Crop_and_resize(data):

    da = transforms.RandomResizedCrop(9, scale = (0.08, 1.0), ratio=(0.75, 1.3333333333333333))
    #size： 用于Resize功能，指定最终得到的图片大小，最终图片大小为9*9
    #scale：用于Crop功能，指定裁剪区域的面积占原图像的面积的比例范围，是一个二元组，如（scale_lower, scale_upper），我们会在[scale_lower, scale_upper]这个区间中随机采样一个值。
    #ratio：用于Crop功能，指定裁剪区域的宽高比范围，是一个二元组，如（ratio_lower,ratio_upper），我们会在[ratio_lower, ratio_upper]这个区间中随机采样一个值。
    data = data.transpose(2, 0, 1)
    x = da(torch.from_numpy(data))
    x = x.numpy()
    x = x.transpose(1, 2, 0)
    return x

def flip_augmentation(data): # arrays tuple 0:(7, 7, 103) 1=(7, 7)
    horizontal = np.random.random() > 0.5 # True
    vertical = np.random.random() > 0.5 # False
    if horizontal:
        data = np.fliplr(data)
    if vertical:
        data = np.flipud(data)
    return data

def crop_augmentation(data): # arrays tuple 0:(7, 7, 103) 1=(7, 7)
    crop_center(data, 3, 4)
    return data


def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2
    img = img[starty:starty+cropy, startx:startx+cropx, :]
    newImg = resize(img, (9, 9))
    return newImg



class Task(object):

    def __init__(self, data, num_classes, shot_num, query_num):
        self.data = data
        self.num_classes = num_classes
        self.support_num = shot_num
        self.query_num = query_num

        class_folders = sorted(list(data))

        class_list = random.sample(class_folders, self.num_classes)

        labels = np.array(range(len(class_list)))

        labels = dict(zip(class_list, labels))

        samples = dict()

        self.support_datas = []
        self.query_datas = []
        self.support_labels = []
        self.query_labels = []
        for c in class_list:
            temp = self.data[c]  # list
            samples[c] = random.sample(temp, len(temp))
            random.shuffle(samples[c])

            self.support_datas += samples[c][:shot_num]
            self.query_datas += samples[c][shot_num:shot_num + query_num]

            self.support_labels += [labels[c] for i in range(shot_num)]
            self.query_labels += [labels[c] for i in range(query_num)]
            # print(self.support_labels)
            # print(self.query_labels)

class FewShotDataset(Dataset):
    def __init__(self, task, split='train'):
        self.task = task
        self.split = split
        self.image_datas = self.task.support_datas if self.split == 'train' else self.task.query_datas
        self.labels = self.task.support_labels if self.split == 'train' else self.task.query_labels

    def __len__(self):
        return len(self.image_datas)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")

class HBKC_dataset(FewShotDataset):
    def __init__(self, *args, **kwargs):
        super(HBKC_dataset, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        image = self.image_datas[idx]
        label = self.labels[idx]
        return image, label

# Sampler
class ClassBalancedSampler(Sampler):
    ''' Samples 'num_inst' examples each from 'num_cl' pool of examples of size 'num_per_class' '''
    # 参数：
    #   num_per_class: 每个类的样本数量
    #   num_cl: 类别数量
    #   num_inst：support set或query set中的样本数量
    #   shuffle：样本是否乱序
    def __init__(self, num_per_class, num_cl, num_inst,shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batch = [[i+j*self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        else:
            batch = [[i+j*self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1

# dataloader
def get_HBKC_data_loader(task, num_per_class=1, split='train',shuffle = True):
    # 参数:
    #   task: 当前任务
    #   num_per_class:每个类别的样本数量，与split有关
    #   split：‘train'或‘test'代表support和querya
    #   shuffle：样本是否乱序
    # 输出：
    #   loader
    dataset = HBKC_dataset(task,split=split)

    if split == 'train':
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.support_num, shuffle=shuffle) # support set
    else:
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.query_num, shuffle=shuffle) # query set

    loader = DataLoader(dataset, batch_size=num_per_class*task.num_classes, sampler=sampler)

    return loader

def classification_map(map, groundTruth, dpi, savePath):

    fig = plt.figure(frameon=False)
    fig.set_size_inches(groundTruth.shape[1]*2.0/dpi, groundTruth.shape[0]*2.0/dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(savePath, dpi = dpi)

    return 0

# def set_noise(data,noise_percentage):
#     class_num = len(data.keys())
#     noise_num = int(200 * noise_percentage)
#     for i in range(class_num):
#         for j in range(noise_num):
#             noise_i = i
#             noise_j = j
#             while (noise_i == i):
#                 noise_i = random.randint(0, class_num-1)  # 生成1到10之间的随机整数
#             while (noise_j == j):
#                 noise_j = random.randint(0, noise_num - 1)
#             data[i][j], data[noise_i][noise_j] = data[noise_i][noise_j], data[i][j]
#             # print(f"{i}类与{noise_i}类第{j}个数据作交换")
#         np.random.shuffle(data[i])
#     return data


def set_noise(data,noise_percentage):
    class_num = len(data.keys())
    noise_num = int(200 * noise_percentage)
    data_new = []
    for i in range(class_num):
        for j in range(200):
            data_new.append(data[i][j])
    np.random.shuffle(data_new)
    for i in range(class_num):
        new_i = i*noise_num
        new_j = (i+1)*noise_num
        data[i][0:noise_num] = data_new[new_i:new_j]
        np.random.shuffle(data[i])
    return data

def gen_prototypes(embeddings, ways, shots, agg_method="mean"):
    assert (
        embeddings.size(0) == ways * shots
    ), "# of embeddings ({}) doesn't match ways ({}) and shots ({})".format(
        embeddings.size(0), ways, shots
    )

    embeddings = embeddings.reshape(ways, shots, -1)
    mean_embeddings = embeddings.mean(dim=1)

    if agg_method == "mean":
        return mean_embeddings

    elif agg_method == "median":
        # Init median as mean
        median_embeddings = torch.unsqueeze(mean_embeddings, dim=1)
        c = 0.5
        for i in range(5):
            errors = median_embeddings - embeddings
            # Poor man's Newton's method
            denom = torch.sqrt(torch.sum(errors ** 2, axis=2, keepdims=True) + c ** 2)
            dw = -torch.sum(errors / denom, axis=1, keepdims=True) / torch.sum(
                1.0 / denom, axis=1, keepdims=True
            )
            median_embeddings += dw
        return torch.squeeze(median_embeddings, dim=1)

    elif (
        agg_method.startswith("cosine")
        or agg_method.startswith("euclidean")
        or agg_method.startswith("abs")
    ):
        epsilon = 1e-6

        if agg_method.startswith("cosine"):
            # Normalize all embeddings to unit vectors
            norm_embeddings = embeddings / (
                torch.norm(embeddings, dim=2, keepdim=True) + epsilon
            )
            # Calculate cosine angle between all support samples in each class: ways x shots x shots
            # Make negative, as higher cosine angle means greater correlation
            cos = torch.bmm(norm_embeddings, norm_embeddings.permute(0, 2, 1))
            attn = (torch.sum(cos, dim=1) - 1) / (shots - 1)
        elif agg_method.startswith("euclidean"):
            # dist: ways x shots x shots
            dist = (
                (embeddings.unsqueeze(dim=2) - embeddings.unsqueeze(dim=1)) ** 2
            ).sum(dim=-1)
            attn = -torch.sum(dist, dim=1) / (shots - 1)
        elif agg_method.startswith("abs"):
            # dist: ways x shots x shots
            dist = (
                torch.abs(embeddings.unsqueeze(dim=2) - embeddings.unsqueeze(dim=1))
            ).sum(dim=-1)
            attn = -torch.sum(dist, dim=1) / (shots - 1)

        # Parse softmax temperature (default=1)
        T = float(agg_method.split("_")[-1]) if "_" in agg_method else 1
        weights = F.softmax(attn / T, dim=1).unsqueeze(dim=2)
        # print(f'weightmax{torch.max(weights)},weightmin{torch.min(weights)}')
        weighted_embeddings = embeddings * weights
        new_features = weighted_embeddings.reshape(-1,160)
        return weighted_embeddings.sum(dim=1),new_features

    else:
        raise NotImplementedError


def gen_prototypes_one(embeddings, ways, shots, agg_method="mean"):
    assert (
        embeddings.size(0) == ways * shots
    ), "# of embeddings ({}) doesn't match ways ({}) and shots ({})".format(
        embeddings.size(0), ways, shots
    )

    embeddings = embeddings.reshape(ways, shots, -1)
    mean_embeddings = embeddings.mean(dim=1)

    if agg_method == "mean":
        return mean_embeddings

    elif agg_method == "median":
        # Init median as mean
        median_embeddings = torch.unsqueeze(mean_embeddings, dim=1)
        c = 0.5
        for i in range(5):
            errors = median_embeddings - embeddings
            # Poor man's Newton's method
            denom = torch.sqrt(torch.sum(errors ** 2, axis=2, keepdims=True) + c ** 2)
            dw = -torch.sum(errors / denom, axis=1, keepdims=True) / torch.sum(
                1.0 / denom, axis=1, keepdims=True
            )
            median_embeddings += dw
        return torch.squeeze(median_embeddings, dim=1)

    elif (
        agg_method.startswith("cosine")
        or agg_method.startswith("euclidean")
        or agg_method.startswith("abs")
    ):
        epsilon = 1e-6

        if agg_method.startswith("cosine"):
            # Normalize all embeddings to unit vectors
            norm_embeddings = embeddings / (
                torch.norm(embeddings, dim=2, keepdim=True) + epsilon
            )
            # Calculate cosine angle between all support samples in each class: ways x shots x shots
            # Make negative, as higher cosine angle means greater correlation
            cos = torch.bmm(norm_embeddings, norm_embeddings.permute(0, 2, 1))
            attn = (torch.sum(cos, dim=1) - 1) / (shots - 1)
        elif agg_method.startswith("euclidean"):
            # dist: ways x shots x shots
            dist = (
                (embeddings.unsqueeze(dim=2) - embeddings.unsqueeze(dim=1)) ** 2
            ).sum(dim=-1)
            attn = -torch.sum(dist, dim=1) / (shots - 1)
        elif agg_method.startswith("abs"):
            # dist: ways x shots x shots
            dist = (
                torch.abs(embeddings.unsqueeze(dim=2) - embeddings.unsqueeze(dim=1))
            ).sum(dim=-1)
            attn = -torch.sum(dist, dim=1) / (shots - 1)

        # Parse softmax temperature (default=1)
        T = float(agg_method.split("_")[-1]) if "_" in agg_method else 1
        weights = F.softmax(attn / T, dim=1).unsqueeze(dim=2)
        # print(f'weightmax{torch.max(weights)},weightmin{torch.min(weights)}')
        weighted_embeddings = embeddings * weights
        return weighted_embeddings.sum(dim=1)

    else:
        raise NotImplementedError


def gen_prototypes_weights(embeddings, ways, shots, agg_method="mean"):
    assert (
        embeddings.size(0) == ways * shots
    ), "# of embeddings ({}) doesn't match ways ({}) and shots ({})".format(
        embeddings.size(0), ways, shots
    )

    embeddings = embeddings.reshape(ways, shots, -1)
    mean_embeddings = embeddings.mean(dim=1)

    if agg_method == "mean":
        return mean_embeddings

    elif agg_method == "median":
        # Init median as mean
        median_embeddings = torch.unsqueeze(mean_embeddings, dim=1)
        c = 0.5
        for i in range(5):
            errors = median_embeddings - embeddings
            # Poor man's Newton's method
            denom = torch.sqrt(torch.sum(errors ** 2, axis=2, keepdims=True) + c ** 2)
            dw = -torch.sum(errors / denom, axis=1, keepdims=True) / torch.sum(
                1.0 / denom, axis=1, keepdims=True
            )
            median_embeddings += dw
        return torch.squeeze(median_embeddings, dim=1)

    elif (
        agg_method.startswith("cosine")
        or agg_method.startswith("euclidean")
        or agg_method.startswith("abs")
    ):
        epsilon = 1e-6

        if agg_method.startswith("cosine"):
            # Normalize all embeddings to unit vectors
            norm_embeddings = embeddings / (
                torch.norm(embeddings, dim=2, keepdim=True) + epsilon
            )
            # Calculate cosine angle between all support samples in each class: ways x shots x shots
            # Make negative, as higher cosine angle means greater correlation
            cos = torch.bmm(norm_embeddings, norm_embeddings.permute(0, 2, 1))
            attn = (torch.sum(cos, dim=1) - 1) / (shots - 1)
        elif agg_method.startswith("euclidean"):
            # dist: ways x shots x shots
            dist = (
                (embeddings.unsqueeze(dim=2) - embeddings.unsqueeze(dim=1)) ** 2
            ).sum(dim=-1)
            attn = -torch.sum(dist, dim=1) / (shots - 1)
        elif agg_method.startswith("abs"):
            # dist: ways x shots x shots
            dist = (
                torch.abs(embeddings.unsqueeze(dim=2) - embeddings.unsqueeze(dim=1))
            ).sum(dim=-1)
            attn = -torch.sum(dist, dim=1) / (shots - 1)

        # Parse softmax temperature (default=1)
        T = float(agg_method.split("_")[-1]) if "_" in agg_method else 1
        weights = F.softmax(attn / T, dim=1).unsqueeze(dim=2)
        # print(f'weightmax{torch.max(weights)},weightmin{torch.min(weights)}')
        weighted_embeddings = embeddings * weights
        weight = weights.reshape(-1,1)
        return weighted_embeddings.sum(dim=1),weight

    else:
        raise NotImplementedError
