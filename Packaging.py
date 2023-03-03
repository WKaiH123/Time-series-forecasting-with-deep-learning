"""
@project: Time series forecasting with deep learning
@autor：Kaihan Wu
@file：Packaging.py
@time：2023-03-02
@vision：1.1
@email: wkaihan168@gmail.com
"""

import csv
import os
import pandas as pd
import torch
from torch.nn.utils import weight_norm
from torch import nn
import torch.optim as optim
import time
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pylab as plt
import matplotlib.pyplot as plt
from torch.autograd import Variable
from window_split import get_batch
from itertools import chain
import torch.nn.functional as F
import causal_convolution_layer
from sklearn.metrics import mean_absolute_error,mean_squared_error,median_absolute_error,r2_score,mean_squared_log_error
import random as rd
from numpy import random

#RNN模型结构
class RNNModel(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(RNNModel, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            # use batch_first for input with another data shape with b first
        )
        # compress output to the same dim as y
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        # Propagate input through LSTM
        out, h_0 = self.rnn(x, h_0)

        out = self.fc(out[:, -1, :])
        return out

#MLP模型结构
class MLPRegress(nn.Module):
    def __init__(self, input_channels, output_channels, channel_sizes):
        super(MLPRegress, self).__init__()
        self.channel_sizes = channel_sizes
        layers = []
        for i in range(len(channel_sizes)):
            if i == 0:
                self.linear = nn.Linear(input_channels, channel_sizes[i])
                self.init_weights()
                layers += [self.linear, nn.ReLU()]

            else:
                self.linear = nn.Linear(channel_sizes[i - 1], channel_sizes[i])
                self.init_weights()
                layers += [self.linear, nn.ReLU()]
        self.linear = nn.Linear(channel_sizes[-1], output_channels)
        self.init_weights()
        layers += [self.linear]
        self.network = nn.Sequential(*layers)

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        return self.network(x)

#lstm模型结构
class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out[-1, :, :].view(-1, self.hidden_size)

        out = self.fc(h_out)

        return out

#TCN模型结构
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # 第一次卷积
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, 3096, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # 第二次卷积
        self.conv2 = weight_norm(nn.Conv1d(3096, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2,
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        self.relu = nn.ReLU()
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.relu(self.network(x) + x[:, 0, :].unsqueeze(1))

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):  # input_size 输入的不同的时间序列数目
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.downsample = nn.Conv1d(input_size, num_channels[0], 1)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)
        self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y1 = self.relu(self.tcn(x) + x[:, 0, :].unsqueeze(1))
        return self.linear(y1[:, :, -1])

#GRU模型结构
class GRU(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(GRU, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        # Propagate input through LSTM
        out, h_0 = self.gru(x, h_0)
        out = out[:, -1]
        out = self.fc(out)

        return out

#封装好的BLS模型类
class BLSregressor:
    def __init__(self, s, C, NumFea, NumWin, NumEnhan):
        self.s = s
        self.C = C
        self.NumFea = NumFea
        self.NumEnhan = NumEnhan
        self.NumWin = NumWin

    def shrinkage(self, a, b):
        z = np.maximum(a - b, 0) - np.maximum(-a - b, 0)
        return z

    def tansig(self, x):
        return (2 / (1 + np.exp(-2 * x))) - 1

    def pinv(self, A, reg):
        return np.mat(reg * np.eye(A.shape[1]) + A.T.dot(A)).I.dot(A.T)

    def sparse_bls(self, A, b):
        lam = 0.001
        itrs = 50
        AA = np.dot(A.T, A)
        m = A.shape[1]
        n = b.shape[1]
        wk = np.zeros([m, n], dtype='double')
        ok = np.zeros([m, n], dtype='double')
        uk = np.zeros([m, n], dtype='double')
        L1 = np.mat(AA + np.eye(m)).I
        L2 = np.dot(np.dot(L1, A.T), b)
        for i in range(itrs):
            tempc = ok - uk
            ck = L2 + np.dot(L1, tempc)
            ok = self.shrinkage(ck + uk, lam)
            uk += ck - ok
            wk = ok
        return wk

    def fit(self, train_x, train_y):
        train_y = train_y.reshape(-1, 1)
        u = 0
        WF = list()
        for i in range(self.NumWin):
            rd.seed(i + u)
            WeightFea = 2 * random.randn(train_x.shape[1] + 1, self.NumFea) - 1
            WF.append(WeightFea)
        rd.seed(100)
        WeightEnhan = 2 * random.randn(self.NumWin * self.NumFea + 1, self.NumEnhan) - 1
        H1 = np.hstack([train_x, 0.1 * np.ones([train_x.shape[0], 1])])
        y = np.zeros([train_x.shape[0], self.NumWin * self.NumFea])
        WFSparse = list()
        distOfMaxAndMin = np.zeros(self.NumWin)
        meanOfEachWindow = np.zeros(self.NumWin)
        for i in range(self.NumWin):
            WeightFea = WF[i]
            A1 = H1.dot(WeightFea)
            scaler1 = MinMaxScaler(feature_range=(-1, 1)).fit(A1)
            A1 = scaler1.transform(A1)
            WeightFeaSparse = self.sparse_bls(A1, H1).T
            WFSparse.append(WeightFeaSparse)

            T1 = H1.dot(WeightFeaSparse)
            meanOfEachWindow[i] = T1.mean()
            distOfMaxAndMin[i] = T1.max() - T1.min()
            T1 = (T1 - meanOfEachWindow[i]) / distOfMaxAndMin[i]
            y[:, self.NumFea * i:self.NumFea * (i + 1)] = T1
        H2 = np.hstack([y, 0.1 * np.ones([y.shape[0], 1])])
        T2 = H2.dot(WeightEnhan)
        T2 = self.tansig(T2)
        T3 = np.hstack([y, T2])
        WeightTop = self.pinv(T3, self.C).dot(train_y)
        self.WeightTop = WeightTop
        self.WFSparse = WFSparse
        self.meanOfEachWindow = meanOfEachWindow
        self.distOfMaxAndMin = distOfMaxAndMin
        self.WeightEnhan = WeightEnhan
        return self

    def predict(self, test_x):
        HH1 = np.hstack([test_x, 0.1 * np.ones([test_x.shape[0], 1])])
        yy1 = np.zeros([test_x.shape[0], self.NumWin * self.NumFea])
        for i in range(self.NumWin):
            WeightFeaSparse = self.WFSparse[i]
            TT1 = HH1.dot(WeightFeaSparse)
            TT1 = (TT1 - self.meanOfEachWindow[i]) / self.distOfMaxAndMin[i]
            yy1[:, self.NumFea * i:self.NumFea * (i + 1)] = TT1
        HH2 = np.hstack([yy1, 0.1 * np.ones([yy1.shape[0], 1])])
        TT2 = self.tansig(HH2.dot(self.WeightEnhan))
        TT3 = np.hstack([yy1, TT2])
        NetoutTest = TT3.dot(self.WeightTop)
        NetoutTest = np.array(NetoutTest).reshape(1, -1)
        return NetoutTest

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=False):
        return {
            's': self.s,
            'C': self.C,
            'NumFea': self.NumFea,
            'NumWin': self.NumWin,
            'NumEnhan': self.NumEnhan
        }

#训练模型及参数设定
class models:

    def __init__(self, path_results = './results/',path_bestmodel = './models/',path_prediction = './prediction/',path_figure = './figure/'):
        # 创建保存结果的文件夹
        self.path_results = path_results
        folder = os.path.exists(self.path_results)
        if not folder:
            os.makedirs(self.path_results)
        # 创建保存模型的文件夹
        self.path_bestmodel = path_bestmodel
        folder = os.path.exists(self.path_bestmodel)
        if not folder:
            os.makedirs(self.path_bestmodel)
        # 创建保存预测值的文件夹
        self.path_prediction = path_prediction
        folder = os.path.exists(self.path_prediction)
        if not folder:
            os.makedirs(self.path_prediction)
        # 创建保存拟合图的文件夹
        self.path_figure = path_figure
        folder = os.path.exists(self.path_figure)
        if not folder:
            os.makedirs(self.path_figure)

    # 回归任务的评估指标，pf代表是否打印出来结果，默认不打印
    def calculate(self,y_true, y_predict,pf=False):
        y_true = y_true.cpu().detach().numpy().reshape(-1,1)
        y_predict = y_predict.cpu().detach().numpy().reshape(-1,1)
        # 初始化评估结果
        mse = np.inf
        rmse = np.inf
        mae = np.inf
        r2 = np.NINF
        mad = np.inf
        mape = np.inf
        r2_adjusted = np.NINF
        rmsle = np.inf
        pcorr = np.NINF
        # try except 的原因是有时候有些结果不适合用某种评估指标
        try:
            mse = mean_squared_error(y_true, y_predict)
        except:
            pass
        try:
            rmse = np.sqrt(mean_squared_error(y_true, y_predict))
        except:
            pass
        try:
            mae = mean_absolute_error(y_true, y_predict)
        except:
            pass
        try:
            r2 = r2_score(y_true, y_predict)
        except:
            pass
        try:
            mad = median_absolute_error(y_true, y_predict)
        except:
            pass
        try:
            mape = np.mean(np.abs((y_true - y_predict) / y_true)) * 100
        except:
            pass
        # try:
        #     if(n>p):
        #         r2_adjusted = 1-((1-r2)*(n-1))/(n-p-1)
        # except:
        #     pass
        try:
            rmsle = np.sqrt(mean_squared_log_error(y_true,y_predict))
            rmsle = round(rmsle, 4)
        except:
            pass
        try:
            pcorr = np.corrcoef(y_predict.reshape(1,-1), y_true.reshape(1,-1))[0][1]
        except:
            pass
        if(pf):
            try:
                print('MSE: ', mse)
            except:
                pass
            try:
                print('RMSE: ', rmse)
            except:
                pass
            try:
                print('MAE: ', mae)
            except:
                pass
            try:
                print('R2: ', r2)
            except:
                pass
            try:
                print('MAD:', mad)
            except:
                pass
            try:
                print('MAPE:', mape)
            except:
                pass
            # try:
            #     print('R2_Adjusted: ',r2_adjusted)
            # except:
            #     pass
            try:
                print("RMSLE: ",rmsle)
            except:
                pass
            try:
                print("PCORR: ",pcorr)
            except:
                pass
        return mse,rmse,mae,r2,mad,mape,rmsle,pcorr

    #导入实验数据集，并划分训练集和测试集，并进行标准化,最后批量化数据输出
    def load_data(self,data,train_test_rate,time_step,skip):

        self.time_step = time_step
        self.skip = skip
        # 划分训练集与测试集
        # train_test_rate = 0.7
        train_size = int(train_test_rate * len(data))
        train_data = data[:train_size]
        test_data = data[train_size:]


        # 归一化处理  (data_len, features_size)
        self.train_mmin, self.train_mmax = train_data.min(), train_data.max()
        training_data = (train_data - self.train_mmin) / (self.train_mmax - self.train_mmin)
        train_data = torch.from_numpy(training_data.values).float().unsqueeze(0)
        train_targets = torch.from_numpy(training_data.values).float().unsqueeze(1)

        testing_data = (test_data - self.train_mmin) / (self.train_mmax - self.train_mmin)
        test_data = torch.from_numpy(testing_data.values).float().unsqueeze(0)
        test_targets = torch.from_numpy(testing_data.values).float().unsqueeze(1)

        # time_step = 7
        # skip = 1  # 1表示第7天预测第8天
        batch_size = 16
        x_train, y_train = get_batch(train_data[:, :-1], train_targets, self.time_step, self.skip, isTrain=True)
        x_test, y_test = get_batch(test_data, test_targets, self.time_step, self.skip, isTrain=False)
        print('x_train : {}, y_train : {}, x_test : {}, y_test : {}'.format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))

        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)
        self.train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_iter = DataLoader(test_dataset, batch_size=100000, shuffle=False)

    def RNN(self,data,train_test_rate,time_step,skip):   #data:单变量序列，type:DataFrame； train_test_rate：训练集比率,time_step：可观测时间步,skip：预测步长，1为单步预测

        self.load_data(data,train_test_rate,time_step,skip)

        learning_rate = 0.01

        input_size = 1
        hidden_size = 32
        num_layers = 1
        num_classes = 1

        rnn = RNNModel(num_classes, input_size, hidden_size, num_layers)
        hidden_prev = torch.zeros(1, 16, hidden_size)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

        clip = 0.7
        epochs = 100

        path_a = self.path_bestmodel + 'RNN_' + str(time_step) + '-' + str(skip) + '_best.pkl'  # 最好评价结果的模型
        path_b = self.path_results + 'RNN_'+str(time_step)+'-'+str(skip)+'_best.csv'   #最好评价结果的路径
        path_c = self.path_prediction + 'RNN_' + str(time_step) + '-' + str(skip) + '_best_data.csv'  # 最好预测值与真实值的路径
        path_d = self.path_figure + 'RNN_' + str(time_step) + '-' + str(skip) + '_fitting.jpg'  # 拟合图保存的路径
        best_result = 10 * 10 ** 30
        with open(path_b, "a", encoding="utf-8", newline="") as f:
            f = csv.writer(f)
            f.writerow(['epoch', 'RMSE','MAE', 'R2','MAD','MAPE','RMSLE','pcorr'])
        for epoch in range(1, epochs + 1):
            rnn.train()
            total_loss = 0
            for inp, target in self.train_iter:
                optimizer.zero_grad()
                output = rnn(inp.transpose(1, 2))  # 将[1,1,time_step]转化为[1,time_step,1]

                output = output * (self.train_mmax - self.train_mmin) + self.train_mmin
                target = target * (self.train_mmax - self.train_mmin) + self.train_mmin

                loss = F.mse_loss(output, target)
                loss.backward()
                total_loss += loss
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(rnn.parameters(), clip)
                optimizer.step()
            if epoch % 1 == 0:
                rnn.eval()
                for inp_test, inp_target in self.test_iter:
                    output_test = rnn(inp_test.transpose(1, 2))

                    output_test = output_test * (self.train_mmax - self.train_mmin) + self.train_mmin
                    inp_target = inp_target * (self.train_mmax - self.train_mmin) + self.train_mmin

                    test_loss = F.mse_loss(output_test, inp_target)

                    MSE, RMSE, MAE, R_score, MAD, MAPE, RMSLE, pcorr = self.calculate(inp_target,output_test)
                    # inp_target_mean = inp_target.mean().item()
                    # RMSE = sqrt(test_loss)
                    # MAE = abs(output_test - inp_target).sum() / len(inp_target)
                    # MAPE = (abs((output_test - inp_target) / inp_target)).sum() / len(inp_target)
                    # SSR = ((output_test - inp_target_mean) ** 2).sum()
                    # SSE = ((inp_target - output_test) ** 2).sum()
                    # SST = SSR + SSE
                    # R_score = SSR / SST
                    # pcorr = np.corrcoef(output_test.cpu().detach().numpy().reshape(1, -1),
                    #                     inp_target.cpu().detach().numpy().reshape(1, -1))[0][1]
                #print(f"Train Epoch: {epoch}\t train_loss: {loss}\t test_loss: {test_loss.item()}\t RMSE: {RMSE}\t MAPE: {MAPE}\t R_score: {R_score}\t PCORR: {pcorr}")
                if RMSE < best_result:
                    with open(path_b, "a", encoding="utf-8", newline="") as f:
                        f = csv.writer(f)
                        f.writerow([epoch, RMSE, MAE, R_score, MAD, MAPE, RMSLE, pcorr])
                    # 保存整个网络
                    torch.save(rnn.state_dict(), path_a)
                    # 保存预测值和真实值
                    real = inp_target.cpu().detach().numpy()
                    real = list(chain.from_iterable(real))
                    predict = output_test.cpu().detach().numpy()
                    predict = list(chain.from_iterable(predict))
                    data = {'real':real, 'predict':predict}
                    frame = pd.DataFrame(data)
                    frame.to_csv(path_c, index=False)
                    best_result = RMSE
        plt.figure(figsize=(8, 6))
        test = pd.read_csv(path_c)
        plt.plot(test['real'], 'r', linestyle='--')
        plt.plot(test['predict'])
        plt.legend(['True', 'Predict'])
        plt.savefig(path_d)
        plt.show()
        print('实验结果已保存')

    def MLP(self,data,train_test_rate,time_step,skip): #data:单变量序列，type:DataFrame； train_test_rate：训练集比率,time_step：可观测时间步,skip：预测步长，1为单步预测

        self.load_data(data,train_test_rate,time_step,skip)


        input_channels = time_step
        output_channels = 1
        channel_sizes = [128, 256, 512, 64]
        mlp = MLPRegress(input_channels, output_channels, channel_sizes)
        lr = 0.02
        optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)

        import torch.nn.functional as F
        from math import sqrt
        import csv
        clip = 0.7
        epochs = 100

        path_a = self.path_bestmodel + 'MLP_' + str(time_step) + '-' + str(skip) + '_best.pkl'  # 最好评价结果的模型
        path_b = self.path_results + 'MLP_'+str(time_step)+'-'+str(skip)+'_best.csv'   #最好评价结果的路径
        path_c = self.path_prediction + 'MLP_' + str(time_step) + '-' + str(skip) + '_best_data.csv'  # 最好预测值与真实值的路径
        path_d = self.path_figure + 'MLP_' + str(time_step) + '-' + str(skip) + '_fitting.jpg'  # 拟合图保存的路径
        best_result = 10 * 10 ** 30
        with open(path_b, "a", encoding="utf-8", newline="") as f:
            f = csv.writer(f)
            f.writerow(['epoch', 'RMSE','MAE', 'R2','MAD','MAPE','RMSLE','pcorr'])
        for epoch in range(1, epochs + 1):
            mlp.train()
            total_loss = 0
            for inp, target in self.train_iter:
                optimizer.zero_grad()
                output = mlp(inp)[:, :, -1]

                output = output * (self.train_mmax - self.train_mmin) + self.train_mmin
                target = target * (self.train_mmax - self.train_mmin) + self.train_mmin

                loss = F.mse_loss(output, target)
                loss.backward()
                total_loss += loss
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(mlp.parameters(), clip)
                optimizer.step()
            if epoch % 10 == 0:
                mlp.eval()
                for inp_test, inp_target in self.test_iter:
                    output_test = mlp(inp_test)[:, :, -1]

                    output_test = output_test * (self.train_mmax - self.train_mmin) + self.train_mmin
                    inp_target = inp_target * (self.train_mmax - self.train_mmin) + self.train_mmin

                    test_loss = F.mse_loss(output_test, inp_target)
                    MSE, RMSE, MAE, R_score, MAD, MAPE, RMSLE, pcorr = self.calculate(inp_target,output_test)

                #print(f"Train Epoch: {epoch}\t train_loss: {loss}\t test_loss: {test_loss.item()}\t RMSE: {RMSE}\t MAPE: {MAPE}\t R_score: {R_score}\t PCORR: {pcorr}")

                if RMSE < best_result:
                    with open(path_b, "a", encoding="utf-8", newline="") as f:
                        f = csv.writer(f)
                        f.writerow([epoch, RMSE, MAE, R_score, MAD, MAPE, RMSLE, pcorr])
                    # 保存整个网络
                    torch.save(mlp.state_dict(), path_a)
                    # 保存预测值和真实值
                    real = inp_target.cpu().detach().numpy()
                    real = list(chain.from_iterable(real))
                    predict = output_test.cpu().detach().numpy()
                    predict = list(chain.from_iterable(predict))
                    data = {'real':real, 'predict':predict}
                    frame = pd.DataFrame(data)
                    frame.to_csv(path_c, index=False)
                    best_result = RMSE
        plt.figure(figsize=(8, 6))
        test = pd.read_csv(path_c)
        plt.plot(test['real'], 'r', linestyle='--')
        plt.plot(test['predict'])
        plt.legend(['True', 'Predict'])
        plt.savefig(path_d)
        plt.show()
        print('实验结果已保存')
    def LSTM(self,data,train_test_rate,time_step,skip): #data:单变量序列，type:DataFrame； train_test_rate：训练集比率,time_step：可观测时间步,skip：预测步长，1为单步预测
        self.load_data(data,train_test_rate,time_step,skip)

        learning_rate = 0.0001
        input_size = 1
        hidden_size = 128
        num_layers = 2
        num_classes = 1

        lstm = LSTM(num_classes, input_size, hidden_size, num_layers)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

        clip = 0.7

        epochs = 100
        path_a = self.path_bestmodel + 'LSTM_' + str(self.time_step) + '-' + str(self.skip) + '_best.pkl'  # 最好评价结果的模型
        path_b = self.path_results +'LSTM_'+str(self.time_step)+'-'+str(self.skip)+'_best.csv'   #最好评价结果的路径
        path_c = self.path_prediction +'LSTM_' + str(self.time_step) + '-' + str(self.skip) + '_best_data.csv'  # 最好预测值与真实值的路径
        path_d = self.path_figure+'LSTM_' + str(self.time_step) + '-' + str(self.skip) + '_fitting.jpg'  # 拟合图保存的路径
        best_result = 10 * 10 ** 30
        with open(path_b, "a", encoding="utf-8", newline="") as f:
            f = csv.writer(f)
            f.writerow(['epoch', 'RMSE','MAE', 'R2','MAD','MAPE','RMSLE','pcorr'])
        for epoch in range(1, epochs + 1):
            lstm.train()
            total_loss = 0
            for inp, target in self.train_iter:
                optimizer.zero_grad()
                output = lstm(inp.transpose(1, 2))
                output = output * (self.train_mmax - self.train_mmin) + self.train_mmin
                target = target * (self.train_mmax - self.train_mmin) + self.train_mmin
                loss = F.mse_loss(output, target)
                loss.backward()
                total_loss += loss
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(lstm.parameters(), clip)
                optimizer.step()
            if epoch % 1 == 0:
                lstm.eval()
                for inp_test, inp_target in self.test_iter:
                    output_test = lstm(inp_test.transpose(1, 2))

                    output_test = output_test * (self.train_mmax - self.train_mmin) + self.train_mmin
                    inp_target = inp_target * (self.train_mmax - self.train_mmin) + self.train_mmin

                    test_loss = F.mse_loss(output_test, inp_target)
                    MSE, RMSE, MAE, R_score, MAD, MAPE, RMSLE, pcorr = self.calculate(inp_target,output_test)
            #print(f"Train Epoch: {epoch}\t train_loss: {loss}\t test_loss: {test_loss.item()}")
            if RMSE < best_result:
                with open(path_b, "a", encoding="utf-8", newline="") as f:
                    f = csv.writer(f)
                    f.writerow([epoch, RMSE, MAE, R_score, MAD, MAPE, RMSLE, pcorr])
                # 保存整个网络
                torch.save(lstm.state_dict(), path_a)
                # 保存预测值和真实值
                real = inp_target.cpu().detach().numpy()
                real = list(chain.from_iterable(real))
                predict = output_test.cpu().detach().numpy()
                predict = list(chain.from_iterable(predict))
                data = {'real': real, 'predict': predict}
                frame = pd.DataFrame(data)
                frame.to_csv(path_c, index=False)
                best_result = RMSE

        plt.figure(figsize=(8, 6))
        test = pd.read_csv(path_c)
        plt.plot(test['real'], 'r', linestyle='--')
        plt.plot(test['predict'])
        plt.legend(['True', 'Predict'])
        plt.savefig(path_d)
        plt.show()
        print('实验结果已保存')
    def TCN(self,data, train_test_rate, time_step, skip): #data:单变量序列，type:DataFrame； train_test_rate：训练集比率,time_step：可观测时间步,skip：预测步长，1为单步预测

        # 划分训练集与测试集，没有使用的load_data
        # train_test_rate = 0.7
        train_size = int(train_test_rate * len(data))
        train_data = data[:train_size]
        train_targets = train_data          ##targets不做归一化处理，后续计算损失也不用反归一化

        test_data = data[train_size:]
        test_targets = test_data           ##targets不做归一化处理，后续计算损失也不用反归一化

        # 归一化处理  (data_len, features_size)
        train_mmin, train_mmax = train_data.min(), train_data.max()
        training_data = (train_data - train_mmin) / (train_mmax - train_mmin)
        train_data = torch.from_numpy(training_data.values).float().unsqueeze(0)
        train_targets = torch.from_numpy(train_targets.values).float().unsqueeze(1)

        testing_data = (test_data - train_mmin) / (train_mmax - train_mmin)
        test_data = torch.from_numpy(testing_data.values).float().unsqueeze(0)
        test_targets = torch.from_numpy(test_targets.values).float().unsqueeze(1)

        # time_step = 7
        # skip = 1  # 1表示第7天预测第8天
        batch_size = 16
        x_train, y_train = get_batch(train_data[:, :-1], train_targets, time_step, skip, isTrain=True)
        x_test, y_test = get_batch(test_data, test_targets, time_step, skip, isTrain=False)
        print('x_train : {}, y_train : {}, x_test : {}, y_test : {}'.format(x_train.shape, y_train.shape, x_test.shape,
                                                                            y_test.shape))

        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)
        train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_iter = DataLoader(test_dataset, batch_size=100000, shuffle=False)

        nhid = 512  # 隐层神经元个数
        levels = 1  # 层数
        channel_sizes = [nhid] * levels
        k_size = 2  # 卷积核大小
        dropout = 0
        input_channels = 1  # n_features
        n_classes = 1  # label
        tcn = TCN(input_channels, n_classes, channel_sizes, kernel_size=k_size, dropout=dropout)
        lr = 0.3
        optimizer = torch.optim.Adam(tcn.parameters(), lr=lr)
        # 计算参数量
        #total = sum([param.nelement() for param in tcn.parameters()])
        #print("Number of parameter: %.2fM" % (total / 1e6))

        clip = 0.7
        epochs = 100
        path_a = self.path_bestmodel+'TCN_' + str(time_step) + '-' + str(skip) + '_best.pkl'  # 最好评价结果的模型
        path_b = self.path_results+'TCN_'+str(time_step)+'-'+str(skip)+'_best.csv'   #最好评价结果的路径
        path_c = self.path_prediction+'TCN_' + str(time_step) + '-' + str(skip) + '_best_data.csv'  # 最好预测值与真实值的路径
        path_d = self.path_figure + 'TCN_' + str(time_step) + '-' + str(skip) + '_fitting.jpg'  # 拟合图保存的路径
        best_result = 10 * 10 ** 30
        with open(path_b, "a", encoding="utf-8", newline="") as f:
            f = csv.writer(f)
            f.writerow(['epoch', 'RMSE','MAE', 'R2','MAD','MAPE','RMSLE','pcorr'])

        for epoch in range(1, epochs + 1):
            tcn.train()
            total_loss = 0
            for inp, target in train_iter:
                optimizer.zero_grad()
                output = tcn(inp)
                loss = F.mse_loss(output, target)
                loss.backward()
                total_loss += loss
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(tcn.parameters(), clip)
                optimizer.step()
            if epoch % 1 == 0:
                tcn.eval()
                for inp_test, inp_target in test_iter:
                    output_test = tcn(inp_test)
                    test_loss = F.mse_loss(output_test, inp_target)

                    MSE, RMSE, MAE, R_score, MAD, MAPE, RMSLE, pcorr = self.calculate(inp_target, output_test)

                if RMSE < best_result:
                    with open(path_b, "a", encoding="utf-8", newline="") as f:
                        f = csv.writer(f)
                        f.writerow([epoch, RMSE, MAE, R_score, MAD, MAPE, RMSLE, pcorr])
                    # 保存整个网络
                    torch.save(tcn.state_dict(), path_a)
                    # 保存预测值和真实值
                    real = inp_target.cpu().detach().numpy()
                    real = list(chain.from_iterable(real))
                    predict = output_test.cpu().detach().numpy()
                    predict = list(chain.from_iterable(predict))
                    data = {'real': real, 'predict': predict}
                    frame = pd.DataFrame(data)
                    frame.to_csv(path_c, index=False)
                    best_result = RMSE

        plt.figure(figsize=(8, 6))
        test = pd.read_csv(path_c)
        plt.plot(test['real'], 'r', linestyle='--')
        plt.plot(test['predict'])
        plt.legend(['True', 'Predict'])
        plt.savefig(path_d)
        plt.show()
        print('实验结果已保存')
    def GRU(self,data, train_test_rate, time_step, skip): #data:单变量序列，type:DataFrame； train_test_rate：训练集比率,time_step：可观测时间步,skip：预测步长，1为单步预测

        self.load_data(data,train_test_rate,time_step,skip)

        learning_rate = 0.01

        input_size = 1
        hidden_size = 64
        num_layers = 1
        num_classes = 1

        gru = GRU(num_classes, input_size, hidden_size, num_layers)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(gru.parameters(), lr=learning_rate)

        clip = 0.7
        epochs = 100
        count = 0
        path_a = self.path_bestmodel+'GRU_' + str(time_step) + '-' + str(skip) + '_best.pkl'  # 最好评价结果的模型
        path_b = self.path_results+'GRU_' + str(time_step) + '-' + str(skip) + '_best.csv'  # 最好评价结果的路径
        path_c = self.path_prediction+'GRU_' + str(time_step) + '-' + str(skip) + '_best_data.csv'  # 最好预测值与真实值的路径
        path_d = self.path_figure+'GRU_' + str(time_step) + '-' + str(skip) + '_fitting.jpg'  # 拟合图保存的路径
        best_result = 10 * 10 ** 30

        with open(path_b, "a", encoding="utf-8", newline="") as f:
            f = csv.writer(f)
            f.writerow(['epoch', 'RMSE', 'MAE', 'R2', 'MAD', 'MAPE', 'RMSLE', 'pcorr'])
        for epoch in range(1, epochs + 1):
            gru.train()
            total_loss = 0
            for inp, target in self.train_iter:
                optimizer.zero_grad()
                output = gru(inp.transpose(1, 2))
                output = output * (self.train_mmax - self.train_mmin) + self.train_mmin
                target = target * (self.train_mmax - self.train_mmin) + self.train_mmin
                loss = F.mse_loss(output, target)
                loss.backward()
                total_loss += loss
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(gru.parameters(), clip)
                optimizer.step()
            if epoch % 1 == 0:
                gru.eval()
                for inp_test, inp_target in self.test_iter:
                    output_test = gru(inp_test.transpose(1, 2))

                    output_test = output_test * (self.train_mmax - self.train_mmin) + self.train_mmin
                    inp_target = inp_target * (self.train_mmax - self.train_mmin) + self.train_mmin
                    test_loss = F.mse_loss(output_test, inp_target)
                    MSE, RMSE, MAE, R_score, MAD, MAPE, RMSLE, pcorr = self.calculate(inp_target, output_test)

                if RMSE < best_result:
                    with open(path_b, "a", encoding="utf-8", newline="") as f:
                        f = csv.writer(f)
                        f.writerow([epoch, RMSE, MAE, R_score, MAD, MAPE, RMSLE, pcorr])
                    # 保存整个网络
                    torch.save(gru.state_dict(), path_a)
                    # 保存预测值和真实值
                    real = inp_target.cpu().detach().numpy()
                    real = list(chain.from_iterable(real))
                    predict = output_test.cpu().detach().numpy()
                    predict = list(chain.from_iterable(predict))
                    data = {'real': real, 'predict': predict}
                    frame = pd.DataFrame(data)
                    frame.to_csv(path_c, index=False)
                    best_result = RMSE

        plt.figure(figsize=(8, 6))
        test = pd.read_csv(path_c)
        plt.plot(test['real'], 'r', linestyle='--')
        plt.plot(test['predict'])
        plt.legend(['True', 'Predict'])
        plt.savefig(path_d)
        plt.show()
        print('实验结果已保存')
    def XGB(self,data, train_test_rate, time_step, skip): #data:单变量序列，type:DataFrame； train_test_rate：训练集比率,time_step：可观测时间步,skip：预测步长，1为单步预测
        # 不进行标准化处理，因为XGB模型属于树模型结构，样本点的数值缩放不影响分裂点的位置
        # mmin = training_data.min()
        # mmax = training_data.max()
        # series = (training_data - mmin) / (mmax - mmin)


        train_size = int(len(data) * train_test_rate)
        #time_step = 7  # 窗口时间天数
        test_size = len(data) - train_size - time_step
        values = data.values.reshape(-1, 1)

        def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
            n_vars = 1 if type(data) is list else data.shape[1]
            df = pd.DataFrame(data)
            cols = list()
            for i in range(n_in, 0, -1):
                cols.append(df.shift(i))
            for i in range(0, n_out):
                cols.append(df.shift(-i))
            agg = pd.concat(cols, axis=1)
            if dropnan:
                agg.dropna(inplace=True)
            return agg.values

        def train_test_split(data, n_test):
            return data[:-n_test, :], data[-n_test:, :]

        def xgboost_forecast(train, testX, skip):
            train = np.asarray(train)
            # trainX, trainy = train[:, :-1], train[:, -1] # (data_len, time_step)  #有问题，训练集的X,Y，步长始终为1，如果做多步预测，训练集的Y就错了
            trainX, trainy = train[:len(train) - skip, :-1], train[skip:, -1]
            model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)  # 可以修改n_estimators
            model.fit(trainX, trainy)
            yhat = model.predict(testX)
            return yhat,model

        def walk_forward_validation(data, n_train, n_test, skip=0):
            predictions = list()
            size = n_train
            train, test = train_test_split(data, n_test)
            history = [x for x in train]
            testX, testy = test[:len(test) - skip, :-1], test[skip:, -1]
            yhat, xgb = xgboost_forecast(history, testX, skip)
                # print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
            error = mean_squared_error(test[skip:, -1], yhat)
            return error, test[skip:, -1], yhat, train, test,xgb

        series = series_to_supervised(values, n_in=time_step)  # n_in: time_step
        mse, y, yhat, train, test,xgb = walk_forward_validation(series, train_size, test_size,
                                                            skip=skip-1)

        path_a = self.path_bestmodel+'XGB_' + str(time_step) + '-' + str(skip) + '_best.pkl'  # 最好评价结果的模型
        path_b = self.path_results+'XGB_' + str(time_step) + '-' + str(skip) + '_best.csv'  # 最好评价结果的路径
        path_c = self.path_prediction+'XGB_' + str(time_step) + '-' + str(skip) + '_best_data.csv'  # 最好预测值与真实值的路径
        path_d = self.path_figure+'XGB_' + str(time_step) + '-' + str(skip) + '_fitting.jpg'  # 拟合图保存的路径
        best_result = 10 * 10 ** 30

        with open(path_b, "a", encoding="utf-8", newline="") as f:
            f = csv.writer(f)
            f.writerow(['epoch', 'RMSE', 'MAE', 'R2', 'MAD', 'MAPE', 'RMSLE', 'pcorr'])
        # 还原归一化数据
        predictions = np.array(yhat)
        # predictions = predictions * (mmax - mmin) + mmin
        # y = y * (mmax - mmin) + mmin
        # 计算指标
        MSE, RMSE, MAE, R_score, MAD, MAPE, RMSLE, pcorr = self.calculate(torch.from_numpy(y).float(), torch.from_numpy(predictions).float())


        with open(path_b, "a", encoding="utf-8", newline="") as f:
            f = csv.writer(f)
            f.writerow([1, RMSE, MAE, R_score, MAD, MAPE, RMSLE, pcorr])
        # 保存整个网络
        torch.save(xgb, path_a)
        # 保存预测值和真实值

        real = y
        predict = predictions
        data = {'real': real, 'predict': predict}
        frame = pd.DataFrame(data)
        frame.to_csv(path_c, index=False)

        plt.figure(figsize=(8, 6))
        test = pd.read_csv(path_c)
        plt.plot(test['real'], 'r', linestyle='--')
        plt.plot(test['predict'])
        plt.legend(['True', 'Predict'])
        plt.savefig(path_d)
        plt.show()
        print('实验结果已保存')
    def ARIMA(self,data, train_test_rate, time_step, skip): #data:单变量序列，type:DataFrame； train_test_rate：训练集比率,time_step：可观测时间步,skip：预测步长，1为单步预测

        path_a = self.path_bestmodel+'ARIMA_' + str(time_step) + '-' + str(skip) + '_best.pkl'  # 最好评价结果的模型
        path_b = self.path_results+'ARIMA_' + str(time_step) + '-' + str(skip) + '_best.csv'  # 最好评价结果的路径
        path_c = self.path_prediction+'ARIMA_' + str(time_step) + '-' + str(skip) + '_best_data.csv'  # 最好预测值与真实值的路径
        path_d = self.path_figure+'ARIMA_' + str(time_step) + '-' + str(skip) + '_fitting.jpg'  # 拟合图保存的路径

        # 标准化处理
        mmin = data.min()
        mmax = data.max()
        training_data = (data - mmin) / (mmax - mmin)  # 标准化处理  (data_len, features_size)
        training_data = training_data.values

        size = int(len(training_data) * train_test_rate)
        skip = skip - 1
        train, test = training_data[0:size], training_data[size + skip + time_step:]

        # 进行ARIMA预测
        predictions = list()
        for t in range(len(test)):
            history = training_data[t + size:t + size + time_step]
            model = ARIMA(history, order=(1, 1, 0))
            model_fit = model.fit()
            output = model_fit.forecast(skip + 1)
            yhat = output[skip]
            predictions.append(yhat)
        predictions = np.array(predictions)
        # 归一化还原
        predictions = predictions * ((mmax - mmin)) + mmin
        test = test * ((mmax - mmin)) + mmin


        with open(path_b, "a", encoding="utf-8", newline="") as f:
            f = csv.writer(f)
            f.writerow(['epoch', 'RMSE', 'MAE', 'R2', 'MAD', 'MAPE', 'RMSLE', 'pcorr'])

        # 计算指标
        MSE, RMSE, MAE, R_score, MAD, MAPE, RMSLE, pcorr = self.calculate(torch.from_numpy(test).float(),
                                                                          torch.from_numpy(predictions).float())

        with open(path_b, "a", encoding="utf-8", newline="") as f:
            f = csv.writer(f)
            f.writerow([1, RMSE, MAE, R_score, MAD, MAPE, RMSLE, pcorr])
        # 保存整个网络
        torch.save(model, path_a)
        # 保存预测值和真实值

        real = test
        predict = predictions
        data = {'real': real, 'predict': predict}
        frame = pd.DataFrame(data)
        frame.to_csv(path_c, index=False)

        plt.figure(figsize=(8, 6))
        test = pd.read_csv(path_c)
        plt.plot(test['real'], 'r', linestyle='--')
        plt.plot(test['predict'])
        plt.legend(['True', 'Predict'])
        plt.savefig(path_d)
        plt.show()
        print('实验结果已保存')
    def BLS(self,data, train_test_rate, time_step, skip): #data:单变量序列，type:DataFrame； train_test_rate：训练集比率,time_step：可观测时间步,skip：预测步长，1为单步预测

        def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
            n_vars = 1 if type(data) is list else data.shape[1]
            df = pd.DataFrame(data)
            cols = list()
            for i in range(n_in, 0, -1):
                cols.append(df.shift(i))
            for i in range(0, n_out):
                cols.append(df.shift(-i))
            agg = pd.concat(cols, axis=1)
            if dropnan:
                agg.dropna(inplace=True)
            return agg.values

        def train_test_split(data, n_test):
            return data[:-n_test, :], data[-n_test:, :]

        def bls_forecast(train, testX, skip):
            train = np.asarray(train)
            # trainX, trainy = train[:, :-1], train[:, -1] # (data_len, time_step)  #有问题，训练集的X,Y，步长始终为1，如果做多步预测，训练集的Y就错了
            trainX, trainy = train[:len(train) - skip, :-1], train[skip:, -1]
            s = 0.8
            c = 2 ** -30
            nf = 6
            nw = 5
            ne = 41
            model = BLSregressor(s=s, C=c, NumFea=nf, NumWin=nw, NumEnhan=ne)
            model.fit(trainX, trainy)
            pred_test = model.predict(testX)
            return pred_test[0],model

        def walk_forward_validation(data, n_train, n_test, skip=0):
            predictions = list()
            size = n_train
            train, test = train_test_split(data, n_test)
            history = [x for x in train]
            testX, testy = test[:len(test) - skip, :-1], test[skip:, -1]
            yhat,bls = bls_forecast(history, testX, skip)


                # print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
            error = mean_squared_error(testy, yhat)
            return error, test[skip:, -1], yhat, train, test, bls

        # # 标准化处理
        # mmin = training_data.min()
        # mmax = training_data.max()
        # series = (training_data - mmin) / (mmax - mmin)


        train_size = int(len(data) * train_test_rate)
        # time_step = 7  # 窗口时间天数
        test_size = len(data) - train_size - time_step
        data = data.values.reshape(-1, 1)
        series = series_to_supervised(data, n_in=time_step)  # n_in: time_step

        path_a = self.path_bestmodel+'BLS_' + str(time_step) + '-' + str(skip) + '_best.pkl'  # 最好评价结果的模型
        path_b = self.path_results+'BLS_' + str(time_step) + '-' + str(skip) + '_best.csv'  # 最好评价结果的路径
        path_c = self.path_prediction+'BLS_' + str(time_step) + '-' + str(skip) + '_best_data.csv'  # 最好预测值与真实值的路径
        path_d = self.path_figure+'BLS_' + str(time_step) + '-' + str(skip) + '_fitting.jpg'  # 拟合图保存的路径
        best_result = 10 * 10 ** 30
        epochs = 100
        with open(path_b, "a", encoding="utf-8", newline="") as f:
            f = csv.writer(f)
            f.writerow(['epoch', 'RMSE', 'MAE', 'R2', 'MAD', 'MAPE', 'RMSLE', 'pcorr'])
        for epoch in range(1, epochs + 1):
            mse, y, yhat, train, test, bls = walk_forward_validation(series, train_size, test_size, skip=skip - 1)
            # 还原归一化数据
            predictions = np.array(yhat)
            # predictions = predictions * (mmax - mmin) + mmin
            # y = y * (mmax - mmin) + mmin
            # 计算指标
            MSE, RMSE, MAE, R_score, MAD, MAPE, RMSLE, pcorr = self.calculate(torch.from_numpy(y).float(),
                                                                              torch.from_numpy(predictions).float())
            if RMSE < best_result:
                with open(path_b, "a", encoding="utf-8", newline="") as f:
                    f = csv.writer(f)
                    f.writerow([epoch, RMSE, MAE, R_score, MAD, MAPE, RMSLE, pcorr])
                # 保存整个网络
                torch.save(bls, path_a)
                # 保存预测值和真实值
                real = y
                predict = predictions
                pred = {'real': real, 'predict': predict}
                frame = pd.DataFrame(pred)
                frame.to_csv(path_c, index=False)
                best_result = RMSE



        plt.figure(figsize=(8, 6))
        test = pd.read_csv(path_c)
        plt.plot(test['real'], 'r', linestyle='--')
        plt.plot(test['predict'])
        plt.legend(['True', 'Predict'])
        plt.savefig(path_d)
        plt.show()
        print('实验结果已保存')