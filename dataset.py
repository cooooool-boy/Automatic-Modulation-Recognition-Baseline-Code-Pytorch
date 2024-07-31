from torch.utils.data import Dataset
import torch
import numpy as np
import pickle
from torch.utils.data import DataLoader
from collections import Counter
import h5py
import pandas as pd

def loadRML2016(path, test=None, index=0):
    Xd = pickle.load(open(path, 'rb'), encoding='latin')
    np.random.seed(3407)  
    snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
    X_train = []
    lbl_train = []
    X_val = []
    lbl_val = []
    X_test = []
    lbl_test = []

    # SNR:[-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    test_SNR = []
    test_SNR.append(snrs[0])
    test_SNR.append(snrs[-1])
    
    if test == True:
        snr = snrs[index]
        for mod in mods:
            # 对每个信噪比的数据均匀切分
            temp = Xd[(mod, snr)]
            label = (mod, snr)
            # 对预处理好的数据进行打包，并按6:2:2划分数据，制作成投入网络训练的格式
            n_examples = temp.shape[0]
            n_train = int(n_examples * 0.6)
            n_val = int(n_examples * 0.2)
            train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
            val_idx = np.random.choice(list(set(range(0, n_examples)) - set(train_idx)), size=int(n_val), replace=False)
            test_idx = list(set(range(0, n_examples)) - set(train_idx) - set(val_idx))

            X_train.append(temp[train_idx])
            X_val.append(temp[val_idx])
            X_test.append(temp[test_idx])

            for i in range(int(n_train)):
                lbl_train.append(label)
            for i in range(int(n_train), int(n_train + n_val)):
                lbl_val.append(label)
            for i in range(int(n_train + n_val), int(n_examples)):
                lbl_test.append(label)
                
    else:
        for mod in mods:
            for snr in snrs:
                # 对每个模式下每个信噪比的数据均匀切分
                temp = Xd[(mod, snr)]
                label = (mod, snr)
                # 对预处理好的数据进行打包，并按6:2:2划分数据，制作成投入网络训练的格式
                n_examples = temp.shape[0]
                n_train = int(n_examples * 0.6)
                n_val = int(n_examples * 0.2)
                train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
                val_idx = np.random.choice(list(set(range(0, n_examples)) - set(train_idx)), size=int(n_val), replace=False)
                test_idx = list(set(range(0, n_examples)) - set(train_idx) - set(val_idx))

                X_train.append(temp[train_idx])
                X_val.append(temp[val_idx])
                X_test.append(temp[test_idx])

                for i in range(int(n_train)):
                    lbl_train.append(label)
                for i in range(int(n_train), int(n_train + n_val)):
                    lbl_val.append(label)
                for i in range(int(n_train + n_val), int(n_examples)):
                    lbl_test.append(label)
                
    X_train = np.vstack(X_train)
    X_val = np.vstack(X_val)
    X_test = np.vstack(X_test)

    # 得到训练数据和测试数据的类型标签
    Y_train = np.array(list(map(lambda x: mods.index(lbl_train[x][0]), np.arange(0, len(lbl_train)))))
    Y_val = np.array(list(map(lambda x: mods.index(lbl_val[x][0]), np.arange(0, len(lbl_val)))))
    Y_test = np.array(list(map(lambda x: mods.index(lbl_test[x][0]), np.arange(0, len(lbl_test)))))
    
    # 得到训练数据和测试数据的信噪比标签
    snr_train = np.array(list(map(lambda x: snrs.index(lbl_train[x][1]), np.arange(0, len(lbl_train)))))
    snr_val = np.array(list(map(lambda x: snrs.index(lbl_val[x][1]), np.arange(0, len(lbl_val)))))
    snr_test = np.array(list(map(lambda x: snrs.index(lbl_test[x][1]), np.arange(0, len(lbl_test)))))
    
    return (X_train, Y_train, snr_train), (X_val, Y_val, snr_val), (X_test, Y_test, snr_test)

def loadRML2016B(path, test=None, index=0):
    Xd = pickle.load(open(path, 'rb'), encoding='iso-8859-1')
    np.random.seed(3407)  
    mods, snrs = [sorted(list(set([k[j] for k in Xd.keys()]))) for j in [0,1]]
    X_train = []
    lbl_train = []
    X_val = []
    lbl_val = []
    X_test = []
    lbl_test = []

    # SNR:[-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    test_SNR = []
    test_SNR.append(snrs[0])
    test_SNR.append(snrs[-1])
    
    if test == True:
        snr = snrs[index]
        for mod in mods:
            # 对每个信噪比的数据均匀切分
            temp = Xd[(mod, snr)]
            label = (mod, snr)
            # 对预处理好的数据进行打包，并按6:2:2划分数据，制作成投入网络训练的格式
            n_examples = temp.shape[0]
            n_train = int(n_examples * 0.6)
            n_val = int(n_examples * 0.2)
            train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
            val_idx = np.random.choice(list(set(range(0, n_examples)) - set(train_idx)), size=int(n_val), replace=False)
            test_idx = list(set(range(0, n_examples)) - set(train_idx) - set(val_idx))

            X_train.append(temp[train_idx])
            X_val.append(temp[val_idx])
            X_test.append(temp[test_idx])

            for i in range(int(n_train)):
                lbl_train.append(label)
            for i in range(int(n_train), int(n_train + n_val)):
                lbl_val.append(label)
            for i in range(int(n_train + n_val), int(n_examples)):
                lbl_test.append(label)
    else:
        for mod in mods:
            for snr in snrs:
                # 对每个模式下每个信噪比的数据均匀切分
                temp = Xd[(mod, snr)]
                label = (mod, snr)
                # 对预处理好的数据进行打包，并按6:2:2划分数据，制作成投入网络训练的格式
                n_examples = temp.shape[0]
                n_train = int(n_examples * 0.6)
                n_val = int(n_examples * 0.2)
                train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
                val_idx = np.random.choice(list(set(range(0, n_examples)) - set(train_idx)), size=int(n_val), replace=False)
                test_idx = list(set(range(0, n_examples)) - set(train_idx) - set(val_idx))

                X_train.append(temp[train_idx])
                X_val.append(temp[val_idx])
                X_test.append(temp[test_idx])

                for i in range(int(n_train)):
                    lbl_train.append(label)
                for i in range(int(n_train), int(n_train + n_val)):
                    lbl_val.append(label)
                for i in range(int(n_train + n_val), int(n_examples)):
                    lbl_test.append(label)
    
    X_train = np.vstack(X_train)
    X_val = np.vstack(X_val)
    X_test = np.vstack(X_test)

    # 得到训练数据和测试数据的类型标签
    Y_train = np.array(list(map(lambda x: mods.index(lbl_train[x][0]), np.arange(0, len(lbl_train)))))
    Y_val = np.array(list(map(lambda x: mods.index(lbl_val[x][0]), np.arange(0, len(lbl_val)))))
    Y_test = np.array(list(map(lambda x: mods.index(lbl_test[x][0]), np.arange(0, len(lbl_test)))))
    
    # 得到训练数据和测试数据的信噪比标签
    snr_train = np.array(list(map(lambda x: snrs.index(lbl_train[x][1]), np.arange(0, len(lbl_train)))))
    snr_val = np.array(list(map(lambda x: snrs.index(lbl_val[x][1]), np.arange(0, len(lbl_val)))))
    snr_test = np.array(list(map(lambda x: snrs.index(lbl_test[x][1]), np.arange(0, len(lbl_test)))))

    return (X_train, Y_train, snr_train), (X_val, Y_val, snr_val), (X_test, Y_test, snr_test)

def loadRML2018(path, test=None, index=0):
    f = h5py.File(path,'r')
    np.random.seed(3407)
    X = f['X'][:,:,:]  # ndarray(2555904*1024*2),shape
    Y = f['Y'][:,:]  # ndarray(2M*24),class
    Z = f['Z'][:]  # ndarray(2M*1),SNR
    
    # 将ONE-HOT改成数字
    Y = np.argmax(Y, axis=1)
    
    n_examples = X.shape[0]
    n_train = int(n_examples * 0.6)
    n_val = int(n_examples * 0.2)

    train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
    val_idx = np.random.choice(list(set(range(0, n_examples)) - set(train_idx)), size=int(n_val), replace=False)
    test_idx = list(set(range(0, n_examples)) - set(train_idx) - set(val_idx))

    X_train = X[train_idx]
    Y_train = Y[train_idx]
    snr_train = Z[train_idx]
    X_val = X[val_idx]
    Y_val = Y[val_idx]
    snr_val = Z[val_idx]
    X_test = X[test_idx]
    Y_test = Y[test_idx]
    snr_test = Z[test_idx]

    X_train = np.transpose(np.array(X_train), (0, 2, 1))
    X_test = np.transpose(np.array(X_test), (0, 2, 1))
    X_val= np.transpose(np.array(X_val), (0, 2, 1))
    
    print(X_train.shape)
    
    return (X_train, Y_train, snr_train), (X_val, Y_val, snr_val), (X_test, Y_test, snr_test)

class Getdata_RML2018(Dataset):
    def __init__(self, data, label, transform = None):
        super().__init__()
        self.X = data
        self.lbl = label
        self.transform = transform
        print("shape of all data:", self.X.shape)
        
    def __getitem__(self, index):
        x = self.X[index]
        x = torch.from_numpy(x)
        x = x.unsqueeze(0)
        if self.transform is not None:
            x = self.transform(x)
        y = self.lbl[index]
        return x, y
        
    def __len__(self):
        return(self.X.shape[0])

class Getdata_RML2016A(Dataset):
    def __init__(self, data, label, transform = None):
        super().__init__()
        self.X = data
        self.lbl = label
        self.transform = transform
        print("shape of all data:", self.X.shape)
        
    def __getitem__(self, index):
        x = torch.from_numpy(self.X[index])
        x = x.unsqueeze(0)
        if self.transform is not None:
            x = self.transform(x)
        y = self.lbl[index]
        return x, y
        
    def __len__(self):
        return(self.X.shape[0])
    
class Getdata_RML2016A_snr(Dataset):
    def __init__(self, data, label, snr, transform = None):
        super().__init__()
        self.X = data
        self.lbl = label
        self.snr = snr
        self.transform = transform
        print("shape of all data:", self.X.shape)
        
    def __getitem__(self, index):
        x = torch.from_numpy(self.X[index])
        x = x.unsqueeze(0)
        if self.transform is not None:
            x = self.transform(x)
        y = self.lbl[index]
        snr = self.snr[index]
        return x, y, snr
        
    def __len__(self):
        return(self.X.shape[0])
    
# if __name__ == '__main__':
    # (X_train, Y_train, snr_train), (X_val, Y_val, snr_val), (X_test, Y_test, snr_test) = loadRML2016("./signal_data/RML2016abc/RML2016.10a_dict.pkl")
    # print(Counter(snr_train))
    # train_dataset = Getdata_RML2016A_OODversion(X_train, Y_train, snr_train)
    # train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    # print(Counter(train_dataset.dlabels))
    # test_dataset = Getdata_RML2016A_OODversion(X_test, Y_test, snr_test)
    # test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    
    # for idx, (data, target, dy, sy, _) in enumerate(train_dataloader):
    #     print(dy)