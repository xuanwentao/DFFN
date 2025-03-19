import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import scipy.io as sio
import numpy as np
import random
from helper import *
from net import Ours
import logging
import time
Stime = time.time()
seed = 1
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = 'Samson'
if dataset == 'Samson':
    image_file = r'Samson_dataset.mat'
    P, L, col = 3, 156, 95
    pixel = col**2
    LR, EPOCH, batch_size = 1e-3, 1500, 1
    step_size, gamma = 245, 0.9
    a, b,c= 1, 0.01, 0.001
    w= 0.5
    k,p=5,2
    weight_decay_param = 1e-6
data = sio.loadmat(image_file)


HSI = torch.from_numpy(data["Y"])
GT = torch.from_numpy(data["A"]) 

M_true = data['M']

band_Number = HSI.shape[0]
endmember_number, pixel_number = GT.shape

HSI = torch.reshape(HSI, (L, col, col))
GT = torch.reshape(GT, (P, col, col))
Y = abu_similarity(HSI,w)
Y = Y.float()
Np = col*col
model = 'Ours'
if model == 'Ours':
    net = Ours(L,P, Np,k,p)
else:
    logging.error("So such model in our zoo!")

MSE = torch.nn.MSELoss(size_average=True)
    
loss_func = nn.MSELoss(size_average=True,reduce=True,reduction='mean')
optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=weight_decay_param)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

best_loss =1
for epoch in range(EPOCH):   
    scheduler.step()
    x = HSI.unsqueeze(0).cuda() 
    Y = Y.cuda()
    net.train().cuda()
    end, abu, re1,re2 = net(Y)        

    re1 = torch.reshape(re1,(L,col,col))
    re1 = re1.unsqueeze(0)
    re2 = torch.reshape(re2,(L,col,col))
    abu = torch.reshape(abu,(P,col,col))
    
    abu_neg_error = torch.mean(torch.relu(-abu))
    abu_sum_error = torch.mean((torch.sum(abu, dim=0) - 1) ** 2)
    abu_loss = abu_neg_error + abu_sum_error
    
    de_loss = reconstruction_SADloss(re1, re2)  
    re_loss = reconstruction_SADloss(x, re2)  

    total_loss = a*(re_loss) + b*abu_loss + c*de_loss
    optimizer.zero_grad()

    total_loss.backward()
    optimizer.step()

    
    loss = total_loss.cpu().data.numpy()

    if loss < best_loss:
        state = {
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'loss': loss,
                'best_loss': best_loss,
                'optimizer' : optimizer.state_dict(),
        }
        torch.save(state, "./Samsonresult/Ours/SamsonOursbest_model.pth.tar")
        best_loss = loss
    
    if epoch % 100 == 0:
        print(
            "Epoch:",
            epoch,
            "| total_loss: %.4f" % total_loss.cpu().data.numpy(),
        )

  
checkpoint = torch.load("./Samsonresult/Ours/SamsonOursbest_model.pth.tar")
best_loss = checkpoint['best_loss']
loss = checkpoint['loss']
epoch = checkpoint['epoch']

net.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])

net.eval()
end, abu, re1, re2 = net(Y)

re1 = torch.reshape(re1,(L,Np))
re1 = re1.detach().cpu().numpy()

re2 = torch.reshape(re2,(L,Np))
re2 = re2.detach().cpu().numpy()
end = end.detach().cpu().numpy()
abu = torch.reshape(abu,(P,col,col))
abundance_GT = GT
GT_endmember = M_true

Etime = time.time()
Time = [Etime-Stime]
print(Time)
Para = [a,b,c,w,k,p,LR]
abu, abundance_GT, end, GT_endmember = performance(abu, end, abundance_GT, M_true, col, endmember_number)
sio.savemat('./Samsonresult/Ours/result.mat', {'Aest': end,'abu': abu,'re1':re1,'re2': re2, 'Time':Time,'Para':Para})






