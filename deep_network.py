# called by Net_Trainer.py
import torch.nn as nn
import torch
import random
import numpy as np
import os
import time
import copy
from copy import deepcopy
from scipy.io import savemat
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
from data_loader import get_data_loaders


class DeepNet:
    def __init__(self, args, dataset_train, dataset_test):
        self.train_loader = get_data_loaders(args.batch_size, args.user_num, dataset_train)
        self.test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True)

        self.model_list = []
        self.args = args
        self.loss_fun = nn.CrossEntropyLoss()

        self.model = CNN(self.args)
        for i in range(args.user_num):
            self.model_list.append(CNN(self.args))  # initialized randomly to have better results due to increased diversity of the initialization

        # QAM constellation setup
        self.QAM_16 = np.linspace(-1, 1, 4)  # x or y components of the constellation points
        ave_pow = (8/16) * np.sum(self.QAM_16 ** 2)  # each x component is repeated 4 times, same as y component (so 4+4), and in total we have 16 points
        self.QAM_16 /= np.sqrt(ave_pow)

        self.QAM_64 = np.linspace(-1, 1, 8)  # x or y components of the constellation points
        ave_pow = (16/64) * np.sum(self.QAM_64 ** 2)  # each x component is repeated 8 times, same as y component (so 8+8), and in total we have 64 points
        self.QAM_64 /= np.sqrt(ave_pow)

        self.QAM_4 = np.linspace(-1, 1, 2)  # x or y components of the constellation points
        ave_pow = (4/4) * np.sum(self.QAM_4 ** 2)  # each x component is repeated 2 times, same as y component (so 2+2), and in total we have 4 points
        self.QAM_4 /= np.sqrt(ave_pow)

        self.QAM_2 = np.linspace(-1, 1, 2)  # x  components of the constellation points
        ave_pow = (1/2) * np.sum(self.QAM_2 ** 2)  # each x component is repeated 1 time, and in total we have 2 points
        self.QAM_2 /= np.sqrt(ave_pow)

        self.QAM_8 = np.zeros((2, 8))  # 8 points, each with x and y component
        self.QAM_8[0] = np.array((1+np.sqrt(3), 1, 1, 0, 0, -1, -1, -1-np.sqrt(3)))  # comming from standard 8-QAM constellation configuration
        self.QAM_8[1] = np.array((0, 1, -1, 1+np.sqrt(3), -1-np.sqrt(3), 1, -1, 0))
        ave_pow = np.sum(self.QAM_8 ** 2)/8
        self.QAM_8 /= np.sqrt(ave_pow)

        self.QAM_32 = np.zeros((2, 32))
        idx_range = np.linspace(-1, 1, 6)
        idx = np.arange(6)
        self.QAM_32[0] = np.concatenate((idx_range[1:5], idx_range, idx_range, idx_range, idx_range, idx_range[1:5]))
        self.QAM_32[1, 0:4] = idx_range[0]
        self.QAM_32[1, 4+idx] = idx_range[1]
        self.QAM_32[1, 10+idx] = idx_range[2]
        self.QAM_32[1, 16+idx] = idx_range[3]
        self.QAM_32[1, 22+idx] = idx_range[4]
        self.QAM_32[1, 28:32] = idx_range[5]
        ave_pow = np.sum(self.QAM_32 ** 2)/32
        self.QAM_32 /= np.sqrt(ave_pow)

        self.sqrt_var = 10**(-self.args.noise_level/20)  # noise_level in dB

    def init_model(self):
        return CNN(self.args)

    def train(self):
        print("\nTraining device: ", self.args.device)
        print("Training ...\n")
        start_time = time.time()

        iterPerEpoch = len(self.train_loader["0"])
        loss_log = torch.zeros((self.args.epochs * iterPerEpoch, 1))

        self.optimizer = []
        for model in self.model_list:
            self.optimizer.append(optim.SGD(model.parameters(), lr=self.args.learning_rate))

        cnt = 0
        self.bit_error = 0
        self.cnt_error = 1e-16  # to avoid division by zero
        accuracy_list = []
        bit_error_list = []
        for epoch in range(self.args.epochs):
            for i in range(iterPerEpoch):
                running_loss = []
                for k in range(self.args.user_num):
                    (images, labels) = next(enumerate(self.train_loader[str(k)]))[1]

                    b_x = Variable(images.unsqueeze(1).float())
                    b_y = Variable(labels)

                    net_output = self.model_list[k](b_x)[0]  # net_output dim: (batch, user_num)
                    loss = self.loss_fun(net_output, b_y)

                    self.optimizer[k].zero_grad()
                    loss.backward()
                    self.optimizer[k].step()

                    running_loss.append(loss.item())

                accuracy = 0
                if i % self.args.aggregation_intv == 0:  # aggregate every aggregation_intv iterations
                    if self.args.dynamic_QAM:  # adjust accordingly (this adjusts BER during training iterations)
                        if epoch <= 1:
                            self.args.QAM_size = 16
                        elif epoch == 2:
                            self.args.QAM_size = 8
                        else:
                            self.args.QAM_size = 4

                    self.aggregate_models()  # communication happens here

                if i % (2*self.args.aggregation_intv) == 0:  # for accuracy report
                    accuracy = self.test()
                    accuracy_list.append(accuracy)

                if (i-1) % self.args.aggregation_intv == 0:  # for printing
                    running_loss_mean = float(sum(running_loss) / len(running_loss))
                    loss_log[cnt] = running_loss_mean
                    bit_error_list.append(self.bit_error/self.cnt_error)
                    cnt += 1
                    print(
                        f"Iter {cnt}-{epoch + 1}/{self.args.epochs}   "
                        f"Loss (Train): {running_loss_mean:.4f}    "
                        f"BER: {(self.bit_error/self.cnt_error):.3f}     "
                        f"Accuracy: {accuracy_list[-1]:.3f}     ",
                        flush=True
                    )
                    self.bit_error = 0
                    self.cnt_error = 1e-16  # to avoide division by zero

        if self.args.communication:  # dynamic/fixed communication models
            if self.args.dynamic_QAM:
                QAM = 'd'
            else:
                QAM = str(self.args.QAM_size)
        else:
            QAM = 'NA'  # analog communication model

        end_time = time.time()

        saving_path = self.args.path + 'results/'
        if not os.path.exists(saving_path):
            os.mkdir(saving_path)

        mdic_loss = {'loss': loss_log.numpy()}  # trainig loss
        savemat(saving_path + self.args.name + '_opt_' + str(self.args.user_num) + 'a_32L_' + str(self.args.noise_level) + 'dB_' + QAM + 'Q_even_' + str(self.args.seed_idx) + '.mat', mdic_loss)

        mdic_loss = {'acc': np.array(accuracy_list)}
        savemat(saving_path + self.args.name + '_acc_' + str(self.args.user_num) + 'a_32L_' + str(self.args.noise_level) + 'dB_' + QAM + 'Q_even_' + str(self.args.seed_idx) + '.mat', mdic_loss)

        mdic_loss = {'bit': np.array(bit_error_list)}
        savemat(saving_path + self.args.name + '_bit_' + str(self.args.user_num) + 'a_32L_' + str(self.args.noise_level) + 'dB_' + QAM + 'Q_even_' + str(self.args.seed_idx) + '.mat', mdic_loss)

        self.model.training_flag = True
        print(f"\nTraining time: {end_time - start_time:.4f}")

    def test(self):
        if self.model.training_flag:  # if the model is already trained
            self.model.eval()
        elif os.path.exists(self.args.path + "model_run_" + str(self.args.run_index)):
            self.model = torch.load(self.args.path + "model_run_" + str(self.args.run_index))
            self.model.eval()
        else:
            raise Exception("no trained model found!")

        accuracy = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                test_output, _ = self.model(images)
                pred_y = torch.max(test_output, 1)[1].data.squeeze()
                accuracy += (pred_y == labels).sum().item() / float(labels.size(0))

        accuracy /= i+1
        self.model.train()
        return accuracy

    def aggregate_models(self):
        self.model.training_flag = True
        user_num = self.args.user_num
        learning_rate = self.optimizer[0].param_groups[0]["lr"]

        model_list = []
        for i in range(user_num):
            model_list.append(self.model_list[i].parameters())
        model_list.append(self.model.parameters())

        for param in zip(*model_list):
            ave = 0
            for kk in range(user_num):
                if self.args.communication:  # digital communication
                    ave += self.digital_communication_channel((1/user_num) * (param[kk].data - param[user_num].data))  # averaging at the server
                else:  # analog communication
                    sig = (1/user_num) * (param[kk].data - param[user_num].data)
                    sig_pow = torch.mean(sig ** 2)
                    sqrt_var = 10**(-self.args.noise_level/20) * np.sqrt(sig_pow)  # square root of noise_pow
                    ave += sig + sqrt_var * torch.randn(sig.shape)  # averaging at the server

            param[user_num].data += ave  # ave has -grad not +grad
            for kk in range(user_num):
                param[kk].data = deepcopy(param[user_num].data)  # server sends back the average (the same model) to the agents

    def digital_communication_channel(self, tensor):
        bit_num = 32  # bit precision -- HARD CODED
        uint_tensor = (tensor * 2**(bit_num-1)).to(torch.int32)
        sign_idx = uint_tensor < 0
        uint_tensor[sign_idx] += 2**(bit_num-1)  # to make int to uint
        int_values = []

        VIP = self.args.VIP
        vip_part = ''
        vip_idx = 0
        for val in uint_tensor.flatten():
            val_binary = bin(val.item())[2:].zfill(bit_num)
            if VIP:  # sending important bits with error-free communication (or with low BER)
                vip_idx = self.args.vip_length
                vip_part = val_binary[0:vip_idx]

            val_binary_demodulated = self.demodQAM(self.modQAM(val_binary[vip_idx:], size=self.args.QAM_size))  # communication channel
            int_values.append(int(vip_part + val_binary_demodulated, 2))
            self.bit_error += (sum([p != q for p, q in zip(val_binary[vip_idx:], val_binary_demodulated)]))/32  # /32 is HARD CODED. The division is for calculating the BER
            self.cnt_error += 1  # a counter just for averaging BERs! = number of ber values added into self.bit_error

        int_values = torch.tensor(int_values, dtype=torch.int32).reshape(tensor.shape)  # -- HARD CODED
        int_values[sign_idx] -= 2**(bit_num-1)
        recovered_tensor = (int_values / 2**(bit_num-1)).to(torch.float32)  # -- HARD CODED

        return recovered_tensor

    def modQAM(self, binary_stream, size=16):  # QAM modulator

        bit_size = int(np.log2(size))
        stream_size = len(binary_stream)

        if size == 32:
            modulated_stream = []
            stream_size += 1  # redundant bit, as 32bit - 8vip=24bit is not divisible to 5
            binary_stream = "0" + binary_stream  # redundant bit
            self.QAM_inplace = self.QAM_32
            self.QAM_size = 32
            for i in range(int(stream_size/bit_size)):
                stream_chunk = binary_stream[i*bit_size:(i+1)*bit_size]
                I = self.QAM_32[0, int(stream_chunk, 2)]
                I_n = I + self.sqrt_var * np.random.randn()
                Q = self.QAM_32[1, int(stream_chunk, 2)]
                Q_n = Q + self.sqrt_var * np.random.randn()
                modulated_stream.append((I_n, Q_n))

        elif size == 8:
            modulated_stream = []
            # stream_size += 1  # redundant bit
            # binary_stream = "0" + binary_stream  # redundant bit
            self.QAM_inplace = self.QAM_8
            self.QAM_size = 8
            for i in range(int(stream_size/bit_size)):
                stream_chunk = binary_stream[i*bit_size:(i+1)*bit_size]
                I = self.QAM_8[0, int(stream_chunk, 2)]
                I_n = I + self.sqrt_var * np.random.randn()
                Q = self.QAM_8[1, int(stream_chunk, 2)]
                Q_n = Q + self.sqrt_var * np.random.randn()
                modulated_stream.append((I_n, Q_n))
        elif size == 2:
            modulated_stream = []
            self.QAM_inplace = self.QAM_2
            self.QAM_size = 2
            for i in range(int(stream_size/bit_size)):
                stream_chunk = binary_stream[i*bit_size:(i+1)*bit_size]
                I = self.QAM_inplace[int(stream_chunk, 2)]
                I_n = I + self.sqrt_var * np.random.randn()
                modulated_stream.append((I_n))
        else:
            if size == 64:
                self.QAM_inplace = self.QAM_64
                self.QAM_size = 64
            elif size == 16:
                self.QAM_inplace = self.QAM_16
                self.QAM_size = 16
            elif size == 4:
                self.QAM_inplace = self.QAM_4
                self.QAM_size = 4
            else:
                raise Exception("QAM size is not recognized")

            modulated_stream = []
            for i in range(int(stream_size/bit_size)):
                stream_chunk = binary_stream[i*bit_size:(i+1)*bit_size]
                I = self.QAM_inplace[int(stream_chunk[0:int(bit_size/2)], 2)]
                I_n = I + self.sqrt_var * np.random.randn()
                Q = self.QAM_inplace[int(stream_chunk[int(bit_size/2):], 2)]
                Q_n = Q + self.sqrt_var * np.random.randn()
                modulated_stream.append((I_n, Q_n))

        return modulated_stream

    def demodQAM(self, QAM_stream):  # QAM demodulator
        binary_stream = ''
        if self.QAM_size == 8:
            for item in QAM_stream:
                binary_stream += bin(np.argmin((item[0] - self.QAM_8[0])**2 + (item[1] - self.QAM_8[1])**2))[2:].zfill(int(np.log2(self.QAM_size)))  # why zfill? to ensure the correct length of bit output. For example, if the value is 1, it gives the bit sequence 1 instead of 001 in QAM_8.
            # binary_stream = binary_stream[1:]  # removing redundant bit
        elif self.QAM_size == 32:
            for item in QAM_stream:
                binary_stream += bin(np.argmin((item[0] - self.QAM_32[0])**2 + (item[1] - self.QAM_32[1])**2))[2:].zfill(int(np.log2(self.QAM_size)))
            binary_stream = binary_stream[1:]  # removing redundant bit
        elif self.QAM_size == 2:
            for item in QAM_stream:
                binary_stream += bin(np.argmin((item - self.QAM_inplace)**2))[2:].zfill(int(np.log2(self.QAM_size)))
        else:
            for item in QAM_stream:
                binary_stream += bin(np.argmin((item[0] - self.QAM_inplace)**2))[2:].zfill(int(np.log2(self.QAM_size)/2))  # why zfill? to ensure the correct length of bit output. For example, if the value is 1, it gives the bit sequence 1 instead of 01 in QAM_16.
                binary_stream += bin(np.argmin((item[1] - self.QAM_inplace)**2))[2:].zfill(int(np.log2(self.QAM_size)/2))

        return binary_stream


class CNN(nn.Module):  # DNN (modify it accordingly)
    def __init__(self, args):
        super(CNN, self).__init__()
        self.args = args
        self.training_flag = False

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x    # return x for visualization
