import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Function
import torch.nn.functional as F
import math
from torchsummary import summary
from torch.nn.utils import weight_norm

# every baseline : feature extractor + classifier

# GRU
class GRUModel_FE(nn.Module):
    def __init__(self, classes = 11):
        super(GRUModel_FE, self).__init__()
        self.gru = nn.GRU(input_size=2, hidden_size=128, num_layers=2, batch_first=True)

    def forward(self, x):
        x = x.squeeze(1)
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x = x[:, -1, :]  # Get the last time step's output

        return x
    
class GRUModel(nn.Module):
    def __init__(self, classes=11):
        super(GRUModel, self).__init__()
        self.encoder = GRUModel_FE()
        self.fc = nn.Linear(128, classes)
        
    def forward(self, input):
        x = self.encoder(input)
        x = self.fc(x)

        return x
    
# CGDNN
class GaussianDropout(nn.Module):
    def __init__(self, p):
        super(GaussianDropout, self).__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.p
            return x + noise
        return x

def max_pool2d_same_padding(x, kernel_size, stride):
    input_rows = x.size(2)
    input_cols = x.size(3)
    filter_rows = kernel_size[0]
    filter_cols = kernel_size[1]

    pad_along_height = max((input_rows - 1) * stride + filter_rows - input_rows, 0)
    pad_along_width = max((input_cols - 1) * stride + filter_cols - input_cols, 0)

    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
    return F.max_pool2d(x, kernel_size, stride)

class CGDNN_FE(nn.Module):
    def __init__(self, classes=11):
        super(CGDNN_FE, self).__init__()

        dr = 0.2 # dropout rate (%)
        self.conv1 = nn.Conv2d(1, 50, kernel_size=(1, 6), padding='same')
        self.drop1 = GaussianDropout(dr)
        
        self.conv2 = nn.Conv2d(50, 50, kernel_size=(1, 6), padding='same')
        self.drop2 = GaussianDropout(dr)
        
        self.conv3 = nn.Conv2d(50, 50, kernel_size=(1, 6), padding='same')
        self.drop3 = GaussianDropout(dr)
        
        self.gru = nn.GRU(input_size=512, hidden_size=50, num_layers=1, batch_first=True)
        self.drop4 = GaussianDropout(dr)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x1 = max_pool2d_same_padding(x1, kernel_size=(2, 2), stride=1)
        x1 = self.drop1(x1)
        
        x2 = F.relu(self.conv2(x1))
        x2 = max_pool2d_same_padding(x2, kernel_size=(2, 2), stride=1)
        x2 = self.drop2(x2)
        
        x3 = F.relu(self.conv3(x2))
        x3 = max_pool2d_same_padding(x3, kernel_size=(2, 2), stride=1)
        x3 = self.drop3(x3)
        
        x11 = torch.cat((x1, x3), dim=3)
        x4 = x11.view(x11.size(0), 50, -1)  # Reshape to (batch_size, 50, 472)
        x4, _ = self.gru(x4)
        x4 = self.drop4(x4[:, -1, :])  # Take the output of the last time step
        
        return x4
    
class CGDNN(nn.Module):
    def __init__(self, classes=11):
        super(CGDNN, self).__init__()
        self.dr = 0.3
        self.encoder = CGDNN_FE()
        self.fc1 = nn.Linear(50, 256)
        self.fc2 = nn.Linear(256, classes)  
        
    def forward(self, input):
        x = self.encoder(input)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dr)
        x = self.fc2(x)

        return x    
    
# LSTM
class LSTMModel_FE(nn.Module):
    def __init__(self, classes = 11):
        super(LSTMModel_FE, self).__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=128, num_layers=2, batch_first=True)

    def forward(self, x):
        x = x.squeeze(1)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Get the last time step's output

        return x
    
class LSTMModel(nn.Module):
    def __init__(self, classes=11):
        super(LSTMModel, self).__init__()
        self.encoder = LSTMModel_FE()
        self.fc = nn.Linear(128, classes)
        
    def forward(self, input):
        x = self.encoder(input)
        x = self.fc(x)

        return x
    
# IC-AMCNET
class ICAMC_FE(nn.Module):
    def __init__(self, classes=11):
        super(ICAMC_FE, self).__init__()
        dr = 0.2
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 8), padding='same')
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(1, 4), padding='same')
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(1, 8), padding='same')
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 1))
        self.dropout = nn.Dropout(dr)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(1, 8), padding='same')
        self.flatten = nn.Flatten()

    def forward(self, input):
        x = self.pool1(F.relu(self.conv1(input)))
        x = F.relu(self.conv2(x))
        x = self.pool2(F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = F.relu(self.conv4(x))
        x = self.dropout(x)
        x = self.flatten(x)

        return x

class ICAMC(nn.Module):
    def __init__(self, classes=11):
        super(ICAMC, self).__init__()
        self.dr = 0.2
        self.encoder = ICAMC_FE()
        self.dense = nn.Sequential(
            nn.Linear(64*128, 128),
            nn.ReLU(),
            nn.Dropout(self.dr),
            nn.Linear(128, classes),)
        
    def forward(self, input):
        x = self.encoder(input)
        x = self.dense(x)

        return x
    
# DAE模型
class DAE_FE(nn.Module):
    def __init__(self, classes = 11):
        super(DAE_FE, self).__init__()
        self.dr = 0.3
        self.lstm1 = nn.LSTM(input_size=2, hidden_size=32, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=32, hidden_size=32, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(32, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.fc3 = nn.Linear(16, classes)
        self.decoder = nn.Linear(32, 2)

    def forward(self, x):
        x = x.squeeze(1)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x = F.dropout(x, self.dr)
        x, (s1, c1) = self.lstm2(x)
        s1 = s1.squeeze(0)

        return x, s1
    
class DAE(nn.Module):
    def __init__(self, classes=11):
        super(DAE, self).__init__()
        self.dr = 0.3
        self.encoder = DAE_FE()
        self.fc1 = nn.Linear(32, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.fc3 = nn.Linear(16, classes)
        self.decoder = nn.Linear(32, 2)


    def forward(self, input1):
        x, s1 = self.encoder(input1)
        xc = F.relu(self.fc1(s1))
        xc = self.bn1(xc)
        xc = F.dropout(xc, self.dr)
        xc = F.relu(self.fc2(xc))
        xc = self.bn2(xc)
        xc = F.dropout(xc, self.dr)
        label = self.fc3(xc)
        
        xd = self.decoder(x)
        xd = xd.permute(0, 2, 1)
        xd = xd.unsqueeze(1)

        return label, xd

# MCLDNN模型
class MCLDNN_FE(nn.Module):
    def __init__(self, classes=11):
        super(MCLDNN_FE, self).__init__()
        # SeparateChannel Combined Convolutional Neural Networks
        self.conv1_1 = nn.Conv2d(1, 50, kernel_size=(2, 8), padding='same')
        self.conv1_2 = nn.Conv1d(1, 50, kernel_size=8)
        self.conv1_3 = nn.Conv1d(1, 50, kernel_size=8)
        self.conv2 = nn.Conv2d(50, 50, kernel_size=(1, 8), padding='same')
        self.conv4 = nn.Conv2d(100, 100, kernel_size=(2, 5), padding='valid')
        # LSTM Unit
        self.lstm1 = nn.LSTM(input_size=100, hidden_size=128, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True)

    def forward(self, input1, input2, input3):
        x1 = F.relu(self.conv1_1(input1))
        x2 = F.pad(input2, (0, 7))
        x3 = F.pad(input3, (0, 7))
        x2 = F.relu(self.conv1_2(x2))
        x3 = F.relu(self.conv1_3(x3))
        x2 = x2.unsqueeze(2)
        x3 = x3.unsqueeze(2)
        x = torch.cat([x2, x3], dim=2)
        x = F.relu(self.conv2(x))

        x = torch.cat([x1, x],dim=1)
        x = F.relu(self.conv4(x))
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]  # Only the last time step output

        return x
    
class MCLDNN(nn.Module):
    def __init__(self, classes=11):
        super(MCLDNN, self).__init__()
        self.dr = 0.3
        self.encoder = MCLDNN_FE()
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, classes)

    def forward(self, x):
        input1 = x
        input2 = x[:,:,0,:]
        input3 = x[:,:,1,:]
        x = self.encoder(input1, input2, input3)
        x = F.selu(self.fc1(x))
        x = F.dropout(x, self.dr)
        x = F.selu(self.fc2(x))
        x = F.dropout(x, self.dr)
        x = self.fc3(x)

        return x

# PETCGDNN模型
class PETCGDNN_FE(nn.Module):
    def __init__(self, classes=11):
        super(PETCGDNN_FE, self).__init__()
        self.dr = 0.2  # dropout rate (%)
        self.input = nn.Flatten()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=75, kernel_size=(2, 8), padding=0),
            nn.ReLU(),
            nn.Dropout(self.dr),
            nn.Conv2d(in_channels=75, out_channels=25, kernel_size=(1, 5), padding=0),
            nn.ReLU(),
            nn.Dropout(self.dr),
        )

        self.gru = nn.GRU(input_size=25, hidden_size=128, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(256, 1)
        self.fc2 = nn.Linear(128, 1)
        self.fc3 = nn.Linear(128, classes)
        self.softmax = nn.Softmax(dim=1)
        self.ReLU = nn.ReLU()

    def forward(self, input1, input2, input3):
        input2 = input2.squeeze(dim=1)
        input3 = input3.squeeze(dim=1)
        x1 = self.fc1(self.input(input1))
        cos1 = torch.cos(x1)
        sin1 = torch.sin(x1)

        # shape[batchsize, 128]
        x11 = torch.mul(input2, cos1)
        x12 = torch.mul(input3, sin1)
        x21 = torch.mul(input3, cos1)
        x22 = torch.mul(input2, sin1)

        y1 = torch.add(x11, x12)
        y2 = torch.sub(x21, x22)
        y1 = y1.unsqueeze(1)
        y2 = y2.unsqueeze(1)
        x11 = torch.cat((y1, y2), 1)
        x11 = x11.unsqueeze(1)

        # spatial feature
        x3 = self.conv1(x11)
        x3 = x3.squeeze(2)
        x3 = x3.permute(0, 2, 1)

        # temporal feature
        x4, hidden = self.gru(x3)
        x = x4[:, -1, :]
        
        return x

class PETCGDNN(nn.Module):
    def __init__(self, input_shape=[2, 128], input_shape2=[128], classes=11):
        super(PETCGDNN, self).__init__()
        self.encoder = PETCGDNN_FE()
        self.fc3 = nn.Linear(128, classes)

    def forward(self, input1, input2, input3):
        feat = self.encoder(input1, input2, input3)
        x = self.fc3(feat)
        
        return x

# CLDNN模型
class CLDNN_FE(nn.Module):
    def __init__(self, classes=11):
        super(CLDNN_FE, self).__init__()
        self.dr = 0.5
        self.conv1 = nn.Conv2d(1, 50, kernel_size=(1, 8))
        self.conv2 = nn.Conv2d(50, 50, kernel_size=(1, 8))
        self.conv3 = nn.Conv2d(50, 50, kernel_size=(1, 8))
        self.dropout = nn.Dropout(self.dr)
        self.lstm = nn.LSTM(input_size=488, hidden_size=50, batch_first=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = F.pad(x, (0, 4), value=0)
        x = self.relu(self.conv1(x))
        x1 = self.dropout(x)
        
        x2 = F.pad(x1, (0, 4), value=0)
        x2 = self.relu(self.conv2(x2))
        x2 = self.dropout(x2)
        
        x3 = F.pad(x2, (0, 4), value=0)
        x3 = self.relu(self.conv3(x3))
        x3 = self.dropout(x3)

        xc = torch.cat([x1, x3], dim=3)
        xc = xc.reshape(-1, 50, 488)

        fea, _ = self.lstm(xc)
        fea = fea[:, -1, :]

        return fea

class CLDNN(nn.Module):
    def __init__(self, classes=11):
        super(CLDNN, self).__init__()
        self.dr = 0.5
        self.encoder = CLDNN_FE()
        self.fc1 = nn.Linear(50, 256)
        self.dropout = nn.Dropout(self.dr)
        self.fc2 = nn.Linear(256, classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc1(x)
        X = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

# CLDNN模型
class CLDNN2_FE(nn.Module):
    def __init__(self, classes=11):
        super(CLDNN2_FE, self).__init__()
        self.dr = 0.3
        self.conv1 = nn.Conv2d(1, 256, kernel_size=(1, 3))
        self.conv2 = nn.Conv2d(256, 256, kernel_size=(2, 3))
        self.conv3 = nn.Conv2d(256, 80, kernel_size=(1, 3))
        self.conv4 = nn.Conv2d(80, 80, kernel_size=(1, 3))
        self.dropout = nn.Dropout(self.dr)
        self.lstm = nn.LSTM(input_size=120, hidden_size=50, batch_first=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        
        x = self.relu(self.conv2(x))
        x = self.dropout(x)

        x = self.relu(self.conv3(x))
        x = self.dropout(x)
        
        x = self.relu(self.conv4(x))
        x = self.dropout(x)

        x = x.squeeze(2)

        fea, _ = self.lstm(x)
        fea = fea[:, -1, :]

        return fea

class CLDNN2(nn.Module):
    def __init__(self, classes=11):
        super(CLDNN2, self).__init__()
        self.dr = 0.5
        self.encoder = CLDNN2_FE()
        self.fc1 = nn.Linear(50, 128)
        self.dropout = nn.Dropout(self.dr)
        self.fc2 = nn.Linear(128, classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc1(x)
        X = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

if __name__ == '__main__':
    model = CLDNN().cuda()
    # summary(model, (1, 2, 128))
    num_params = sum(p.numel() for p in model.parameters())
    total_size = 1063659 * 4 / (1024 ** 2)
    print(f"total param: {num_params}")
    print(f"Total Model Parameters Size: {total_size:.2f} MB")