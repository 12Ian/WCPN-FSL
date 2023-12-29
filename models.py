import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.autograd import Function


class FeatureTrasformationLayer(nn.Module):
    def __init__(self, in_dimension, out_dimension):
        super(FeatureTrasformationLayer, self).__init__()
        self.gamma = torch.nn.Parameter(torch.ones(1, num_features, 1, 1) * 0.3)
        self.beta = torch.nn.Parameter(torch.ones(1, num_features, 1, 1) * 0.5)

    def forward(self, x):
        gamma = (1 + torch.randn(1, self.num_features, 1, 1, dtype=self.gamma.dtype,device=self.gamma.device) * softplus(self.gamma)).expand_as(out)
        beta = (torch.randn(1, self.num_features, 1, 1, dtype=self.beta.dtype, device=self.beta.device) * softplus(self.beta)).expand_as(out)
        out = gamma * x + beta

        return out


# 映射网络 将任何 a*b*H 维度 像素块 转换成 a*b*100维度
class Mapping(nn.Module):
    def __init__(self, in_dimension, out_dimension):
        super(Mapping, self).__init__()
        self.preconv = nn.Conv2d(in_dimension, out_dimension, 1, 1, bias=False)
        self.preconv_bn = nn.BatchNorm2d(out_dimension)

    def forward(self, x):
        x = self.preconv(x)
        x = self.preconv_bn(x)
        return x

def conv3x3x3(in_channel, out_channel):
    layer = nn.Sequential(
        nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm3d(out_channel),
        # nn.ReLU(inplace=True)
    )
    return layer


class residual_block(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(residual_block, self).__init__()

        self.conv1 = conv3x3x3(in_channel, out_channel)
        self.conv2 = conv3x3x3(out_channel, out_channel)
        self.conv3 = conv3x3x3(out_channel, out_channel)

    def forward(self, x):  # (1,1,100,9,9)
        x1 = F.relu(self.conv1(x), inplace=True)  # (1,8,100,9,9)  (1,16,25,5,5)
        x2 = F.relu(self.conv2(x1), inplace=True)  # (1,8,100,9,9) (1,16,25,5,5)
        x3 = self.conv3(x2)  # (1,8,100,9,9) (1,16,25,5,5)

        out = F.relu(x1 + x3, inplace=True)  # (1,8,100,9,9)  (1,16,25,5,5)
        return out

# 残差网络 9*9*100 1*1*5 32
class D_Res_3d_CNN(nn.Module):
    def __init__(self, in_channel, out_channel1, out_channel2):
        super(D_Res_3d_CNN, self).__init__()

        self.block1 = residual_block(in_channel, out_channel1)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(4, 2, 2), padding=(0, 1, 1), stride=(4, 2, 2))
        self.block2 = residual_block(out_channel1, out_channel2)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(4, 2, 2), stride=(4, 2, 2), padding=(2, 1, 1))
        self.conv = nn.Conv3d(in_channels=out_channel2, out_channels=32, kernel_size=3, bias=False)

    def forward(self, x):  # x:(400,100,9,9)
        x = x.unsqueeze(1)  # (400,1,100,9,9)
        x = self.block1(x)  # (1,8,100,9,9)
        x = self.maxpool1(x)  # (1,8,25,5,5)
        x = self.block2(x)  # (1,16,25,5,5)
        x = self.maxpool2(x)  # (1,16,7,3,3)
        x = self.conv(x)  # (1,32,5,1,1)
        x = x.view(x.shape[0], -1)  # (1,160)
        #         x = F.relu(x)
        # y = self.classifier(x)
        return x#, y

#############################################################################################################

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

class DomainClassifier(nn.Module):
    def __init__(self):# torch.Size([1, 64, 7, 3, 3])
        super(DomainClassifier, self).__init__() #
        self.layer = nn.Sequential(
            nn.Linear(1024, 1024), #nn.Linear(320, 512), nn.Linear(FEATURE_DIM*CLASS_NUM, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),

        )
        self.domain = nn.Linear(1024, 1) # 512

    def forward(self, x, iter_num):
        coeff = calc_coeff(iter_num, 1.0, 0.0, 10,10000.0)
        # register_hook的作用：即对x求导时，对x的导数进行操作，并且register_hook的参数只能以函数的形式传过去。
        x.register_hook(grl_hook(coeff))
        x = self.layer(x)
        domain_y = self.domain(x)
        return domain_y

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class DomainClassifier_New(nn.Module):
    def __init__(self):# torch.Size([1, 64, 7, 3, 3])
        super(DomainClassifier_New, self).__init__() #

        self.fc = nn.Linear(160, 16)

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('fc1', nn.Linear(160, 80))
        self.domain_classifier.add_module('relu1', nn.ReLU(True))
        self.domain_classifier.add_module('dpt1', nn.Dropout())
        self.domain_classifier.add_module('fc2', nn.Linear(80, 40))
        self.domain_classifier.add_module('relu2', nn.ReLU(True))
        self.domain_classifier.add_module('dpt2', nn.Dropout())
        self.domain_classifier.add_module('fc3', nn.Linear(40, 2))

        self.dcis = nn.Sequential()
        self.dci = {}
        for i in range(16):
            self.dci[i] = nn.Sequential()
            self.dci[i].add_module('fc1', nn.Linear(160, 80))
            self.dci[i].add_module('relu1', nn.ReLU(True))
            self.dci[i].add_module('dpt1', nn.Dropout())
            self.dci[i].add_module('fc2', nn.Linear(80, 40))
            self.dci[i].add_module('relu2', nn.ReLU(True))
            self.dci[i].add_module('dpt2', nn.Dropout())
            self.dci[i].add_module('fc3', nn.Linear(40, 2))
            self.dcis.add_module('dci_'+str(i), self.dci[i])

    def forward(self, source_share, target_share, source_logits, target_logits, alpha=0.0):
        # source = self.fc(source_share)
        # p_source = nn.Softmax(dim=1)(source)
        # target = self.fc(target_share)
        # p_target = nn.Softmax(dim=1)(target)
        s_out = []
        t_out = []
        if self.training == True:
            s_reverse_feature = ReverseLayerF.apply(source_share, alpha)
            t_reverse_feature = ReverseLayerF.apply(target_share, alpha)
            s_domain_output = self.domain_classifier(s_reverse_feature)
            t_domain_output = self.domain_classifier(t_reverse_feature)

            for i in range(16):
                ps = source_logits[:, i].reshape((source_share.shape[0], 1))
                fs = ps * s_reverse_feature
                pt = target_logits[:, i].reshape((target_share.shape[0], 1))
                ft = pt * t_reverse_feature
                outsi = self.dcis[i](fs)
                s_out.append(outsi)
                outti = self.dcis[i](ft)
                t_out.append(outti)
        else:
            s_domain_output = 0
            t_domain_output = 0
            s_out = [0]*16
            t_out = [0]*16
        return s_domain_output, t_domain_output, s_out, t_out

class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        # [[160, 1024], [16, 1024]]
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0/len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]

    def to(self, device):
        super(RandomLayer, self).to(device)
        self.random_matrix = [val.to(device) for val in self.random_matrix]


