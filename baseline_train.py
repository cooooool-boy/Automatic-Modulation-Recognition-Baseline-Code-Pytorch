import numpy as np
import time
from torch.utils.data import Dataset
from torch.utils.data import random_split, DataLoader
import torch
import torch.nn as nn
import sys
from AMR.model import PETCGDNN, MCLDNN, DAE, CLDNN2, ICAMC, GRUModel, CGDNN
from AMR.dataset import Getdata_RML2016A, loadRML2016B, loadRML2016
from AMR.util import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import os
from AMR.scheduler import PolynomialLR
import tqdm
import matplotlib.pyplot as plt

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Training parameters
batchsize = 1024
start_epoch = 0
training_epoch = 150
num_class = 11
classes = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']

"""
=== RML2016 DataLoader === 
"""
start = time.time()

# RML2016.10a_dict.pkl
(X_train, Y_train, snr_train), (X_val, Y_val, snr_val), (X_test, Y_test, snr_test) = loadRML2016("./signal_data/RML2016abc/RML2016.10a_dict.pkl")

train_dataset = Getdata_RML2016A(X_train, Y_train)
train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=0, pin_memory=True)

val_dataset = Getdata_RML2016A(X_val, Y_val)
val_dataloader = DataLoader(val_dataset, batch_size=batchsize, shuffle=True, num_workers=0, pin_memory=True)

test_dataset = Getdata_RML2016A(X_test, Y_test)
test_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True, num_workers=0, pin_memory=True)

end = time.time()
print("load dataset time: {:.3f} s".format(end - start))

"""
=== Model Training ===
"""
model = MCLDNN(classes=num_class).cuda()
for name, param in model.named_parameters():
    if 'weight' in name and param.dim() >= 2:
        nn.init.kaiming_normal_(param.data)

# 定义优化器，损失
optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-3, weight_decay=1e-5)
scheduler = PolynomialLR(optimizer, max_iter=training_epoch, power=0.8)
scaler = torch.cuda.amp.GradScaler()
NUM_ACCUMULATION_STEPS = 8
CrossLoss = nn.CrossEntropyLoss()

correct = torch.zeros(1).squeeze().cuda()
correct_ = list(0. for i in range(num_class))
epochs = []
train_losses = []
train_accs = []
val_losses = []
val_accs = []
best_acc = 0

save_path = './checkpoint/MCLDNN/'

if not os.path.exists(save_path):
    os.makedirs(save_path)
    print("已创建目标文件夹：", save_path)
    
start = time.time()

for epoch in range(start_epoch+1, training_epoch, 1):
    # 模型训练
    model.train()
    with tqdm.tqdm(train_dataloader, unit="batch") as tepoch: # 🌟 1. 定义进度条
        for idx, (data, target) in enumerate(tepoch):# 🌟 2. 设置迭代器")
            tepoch.set_description(f"Epoch {epoch}")  # 🌟 3. 设置开头
            data, target = data.cuda().float(), target.cuda().long()    # Data to device
            with torch.cuda.amp.autocast():
                # fea = model.encoder(data, data1, data2)
                output = model(data)
                #output, xd = model(data) 
                loss1 = CrossLoss(output, target)                                # Calculate loss
                #loss2 = CrossLoss(xd, data)
                losstr = loss1

            scaler.scale(losstr).backward()
            losstr = losstr / NUM_ACCUMULATION_STEPS
            
            if ((idx + 1) % NUM_ACCUMULATION_STEPS == 0) or (idx + 1 == len(train_dataloader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            predict_ = output.argmax(dim=1, keepdim=True)
            correct = predict_.eq(target.view_as(predict_)).sum().item() 

            accuracy = correct/len(data)
            tepoch.set_postfix(loss=losstr.item(), accuracy='{:.3f}'.format(accuracy)) # 🌟 4. 设置结尾

    if (epoch + 1) % 5 == 0:
        scheduler.step()
    epochs.append(epoch)
    train_losses.append(losstr.item())
    train_accs.append(accuracy)
    
    #模型测试
    model.eval()
    all_predicts = torch.empty(0, 1).cuda()
    all_targets = torch.empty(0).cuda()
    with torch.no_grad():
        for _ , (imgs, targets) in enumerate(val_dataloader):

            imgs = imgs.cuda().float()
            targets = targets.cuda().long()

            outputs = model(imgs)

            loss = CrossLoss(outputs, targets)
            predicts = outputs.argmax(dim=1, keepdim=True)
            all_targets = torch.cat([all_targets, targets])
            all_predicts = torch.cat([all_predicts, predicts], dim=0)
    
        correct_ = all_predicts.eq(all_targets.view_as(all_predicts)).sum().item()
        accuracy_ = correct_/float(X_val.shape[0])
        print("val_acc:", accuracy_)
    
    val_losses.append(loss.item())
    val_accs.append(accuracy_)
        
    if accuracy_ > best_acc:
        best_acc = accuracy_
        torch.save(model, save_path + 'model_epoch{}_valAcc_{}.pth'.format(epoch+1, accuracy_))
        
    else:
        torch.save(model, save_path + 'model_epoch{}.pth'.format(epoch+1))

end = time.time()
print("train time: {:.3f} s".format(end - start))

np.savetxt(save_path + 'train_acc.txt',train_accs)
np.savetxt(save_path + 'train_loss.txt',train_losses)
np.savetxt(save_path + 'val_acc.txt',val_accs)
np.savetxt(save_path + 'val_loss.txt',val_losses)
                
plt.subplot(121)
plt.plot(epochs, train_losses, color = 'b')
plt.xlabel('Epoch')
plt.ylabel('Loss')
# 绘制 acc 曲线
plt.subplot(122)
plt.plot(epochs, train_accs, color = 'r')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()