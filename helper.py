import matplotlib.pyplot as plt
import torch
from scipy.spatial.distance import pdist, squareform
import numpy as np
import numpy as np
from scipy.spatial import ConvexHull
import math
#Define Dataset
class MyTrainData(torch.utils.data.Dataset):
  def __init__(self, img, gt, transform=None):
    self.img = img.float()
    self.gt = gt.float()
    self.transform=transform

  def __getitem__(self, idx):
    return self.img,self.gt

  def __len__(self):
    return 1

def euclidean_sim(data):
    norm_data = torch.sum(data ** 2, dim=1).reshape(-1, 1)
    dist_matrix = norm_data - 2 * torch.matmul(data, data.t()) + norm_data.t()
    similarity_matrix = torch.exp(-dist_matrix)
    
    return similarity_matrix

def mat2gray(Y):
    min_va1 = torch.min(Y)
    max_va1 = torch.max(Y)
    data = (Y-min_va1)/(max_va1-min_va1)
    return data

def abu_similarity(data, c):
    y = data
    y = y.squeeze(0)
    b,h,w = y.size()
        
    spa = y.view(b,-1)
    spaT = spa.permute(1,0)
    
    S1 = euclidean_sim(spa)
    S2 = euclidean_sim(spaT)
    y = torch.reshape(y,(b,h*w))
        
    y1 = torch.matmul(S1,y)
    y1 = mat2gray(y1)
    y2 = torch.matmul(y,S2)
    y2 = mat2gray(y2)

    y = c*y1 + (1-c)*y2
    y = mat2gray(y)
    y = torch.reshape(y, (b,h,w))
    y = y.unsqueeze(0)           
    return y


def reconstruction_SADloss(output, target):
    
    _,band, h, w = output.shape
    output = torch.reshape(output, (band, h * w))
    target = torch.reshape(target, (band, h * w))
    abundance_loss = torch.acos(torch.cosine_similarity(output, target, dim=0))
    abundance_loss = torch.mean(abundance_loss)

    return abundance_loss

def abu_neg_sum(abu):
    abu_neg_error = torch.mean(torch.relu(-abu))
    abu_sum_error = torch.mean((torch.sum(abu, dim=0) - 1) ** 2)
    abu_loss = abu_neg_error + abu_sum_error
    return abu_loss

# abundance normalization
def norm_abundance_GT(abundance_input, abundance_GT_input, endmember_number, col):
    abundance_input = abundance_input / (torch.sum(abundance_input, dim=0))
    abundance_input = torch.reshape(
        abundance_input, (endmember_number, col, col)
    )
    abundance_input = abundance_input.cpu().detach().numpy()
    abundance_GT_input = abundance_GT_input / (torch.sum(abundance_GT_input, dim=0))
    abundance_GT_input = abundance_GT_input.cpu().detach().numpy()
    return abundance_input, abundance_GT_input

# endmember normalization
def norm_endmember(endmember_input, endmember_GT, endmember_number):
    for i in range(0, endmember_number):
        endmember_input[:, i] = endmember_input[:, i] / np.max(endmember_input[:, i])
        endmember_GT[:, i] = endmember_GT[:, i] / np.max(endmember_GT[:, i])
    return endmember_input, endmember_GT

# calculate RMSE of abundance
def AbundanceRmse(inputsrc, inputref):
    rmse = np.sqrt(((inputsrc - inputref) ** 2).mean())
    return rmse

# calculate SAD of endmember
def SAD_distance(src, ref):
    cos_sim = np.dot(src, ref) / (np.linalg.norm(src) * np.linalg.norm(ref))
    SAD_sim = np.arccos(cos_sim)
    return SAD_sim


# change the index of abundance and endmember
def arange_A_E(abundance_input, abundance_GT_input, endmember_input, endmember_GT, endmember_number):
    RMSE_matrix = np.zeros((endmember_number, endmember_number))
    SAD_matrix = np.zeros((endmember_number, endmember_number))
    RMSE_index = np.zeros(endmember_number).astype(int)
    SAD_index = np.zeros(endmember_number).astype(int)
    RMSE_abundance = np.zeros(endmember_number)
    SAD_endmember = np.zeros(endmember_number)

    for i in range(0, endmember_number):
        for j in range(0, endmember_number):
            RMSE_matrix[i, j] = AbundanceRmse(
                abundance_input[i, :, :], abundance_GT_input[j, :, :]
            )
            SAD_matrix[i, j] = SAD_distance(endmember_input[:, i], endmember_GT[:, j])

        RMSE_index[i] = np.argmin(RMSE_matrix[i, :])
        SAD_index[i] = np.argmin(SAD_matrix[i, :])
        RMSE_abundance[i] = np.min(RMSE_matrix[i, :])
        SAD_endmember[i] = np.min(SAD_matrix[i, :])

    abundance_input[np.arange(endmember_number), :, :] = abundance_input[
        RMSE_index, :, :
    ]
    endmember_input[:, np.arange(endmember_number)] = endmember_input[:, SAD_index]


    return abundance_input, endmember_input, RMSE_abundance, SAD_endmember

def plot_abundance(abundance_input, abundance_GT_input, endmember_number):
    for i in range(0, endmember_number):

        plt.subplot(2, endmember_number, i + 1)
        plt.imshow(abundance_input[i, :, :], cmap="jet")

        plt.subplot(2, endmember_number, endmember_number + i + 1)
        plt.imshow(abundance_GT_input[i, :, :], cmap="jet")
    plt.show()

# plot endmember
def plot_endmember(endmember_input, endmember_GT, endmember_number):
    for i in range(0, endmember_number):
        plt.subplot(1, endmember_number, i + 1)
        plt.plot(endmember_input[:, i], color="b")
        plt.plot(endmember_GT[:, i], color="r")

    plt.show()

def performance(abu, end, abundance_GT, M_true, col, endmember_number):
    abu, abundance_GT = norm_abundance_GT(abu, abundance_GT, endmember_number, col)
    end, GT_endmember = norm_endmember(end, M_true, endmember_number)

    abu, end, RMSE_abundance, SAD_endmember = arange_A_E(abu, abundance_GT, end, GT_endmember, endmember_number,)

    print("RMSE", RMSE_abundance)
    print("mean_RMSE", RMSE_abundance.mean())
    print("endmember_SAD", SAD_endmember)
    print("mean_SAD", SAD_endmember.mean())
    # plot_endmember(GT_endmember, end, endmember_number)
    # plot_abundance(abundance_GT, abu, endmember_number)
    
    abu = np.reshape(abu,(endmember_number,col*col))
    
    return abu, abundance_GT, end, GT_endmember
    

