# GIANT, ALL MODELS, CODE

# Author: Ljubisa Sehovac
# github: sehovaclj

# code that contains all S2S models used to compute my results. See the conference and journal papers for thorough, detailed, overview


# imporing functions. Uncomment the plt line if you intend to plot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init

import random
import time

import math

#from twilio.rest import Client

from tempfile import TemporaryFile


#########################################################################################

# building the S2S model

class S2S_Model(nn.Module):
    def __init__(self, cell_type, input_size, hidden_size, use_cuda):
        super(S2S_Model, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.cell_type = cell_type

        if self.cell_type not in ['rnn', 'gru', 'lstm']:
            raise ValueError(
                self.cell_type, " is not an appropriate cell type. Please select one of rnn, gru, or lstm.")

        if self.cell_type == 'rnn':
            self.Ecell = nn.RNNCell(self.input_size, self.hidden_size)
            self.Dcell = nn.RNNCell(1, self.hidden_size)

        if self.cell_type == 'gru':
            self.Ecell = nn.GRUCell(self.input_size, self.hidden_size)
            self.Dcell = nn.GRUCell(1, self.hidden_size)

        if self.cell_type == 'lstm':
            self.Ecell = nn.LSTMCell(self.input_size, self.hidden_size)
            self.Dcell = nn.LSTMCell(1, self.hidden_size)

        self.lin_Value = nn.Linear(self.hidden_size, 1)

        self.use_cuda = use_cuda

        self.init()

    # VERY IMPORTANT INIT PARAMS FUNCTIONS***
    # function to intialize weight parameters. Refer to Saxe at al. paper that explains why to use orthogonal init weights
    # note: j was used to see parameter shapes

    def init(self):

        if self.cell_type == 'rnn' or self.cell_type == 'gru':
            #j = []
            for p in self.parameters():
                if p.dim() > 1:
                    init.orthogonal_(p.data, gain=1.0)
                    # j.append(p.data)
                if p.dim() == 1:
                    init.constant_(p.data, 0.0)
                    # j.append(p.data)

        elif self.cell_type == 'lstm':
            #j = []
            for p in self.parameters():
                if p.dim() > 1:
                    init.orthogonal_(p.data, gain=1.0)
                    # j.append(p.data)
                if p.dim() == 1:
                    init.constant_(p.data, 0.0)
                    init.constant_(
                        p.data[self.hidden_size:2*self.hidden_size], 1.0)
                    # j.append(p.data)
        # return j

    def consume(self, x):
        # encoder forward function

        # for rnn and gru
        if self.cell_type == 'rnn' or self.cell_type == 'gru':

            h = torch.zeros(x.shape[0], self.hidden_size)

            if self.use_cuda:
                h = h.cuda()

            for T in range(x.shape[1]):
                h = self.Ecell(x[:, T, :], h)

            pred_Value = self.lin_Value(h)

        # for lstm
        elif self.cell_type == 'lstm':

            h0 = torch.zeros(x.shape[0], self.hidden_size)
            c0 = torch.zeros(x.shape[0], self.hidden_size)

            if self.use_cuda:
                h0 = h0.cuda()
                c0 = c0.cuda()

            h = (h0, c0)

            for T in range(x.shape[1]):
                h = self.Ecell(x[:, T, :], h)

            pred_Value = self.lin_Value(h[0])

        return pred_Value, h

    def predict(self, pred_Value, h, target_length):
        # decoder forward function
        preds = []

        # for rnn and gru
        if self.cell_type == 'rnn' or self.cell_type == 'gru':

            for step in range(target_length):
                h = self.Dcell(pred_Value, h)
                pred_Value = self.lin_Value(h)

                preds.append(pred_Value.unsqueeze(1))

            preds = torch.cat(preds, 1)

        # for lstm
        elif self.cell_type == 'lstm':

            for step in range(target_length):
                h = self.Dcell(pred_Value, h)
                pred_Value = self.lin_Value(h[0])

                preds.append(pred_Value.unsqueeze(1))

            preds = torch.cat(preds, 1)

        return preds


#########################################################################################

# Bahdanau Attention model

# refer to : AuCson github code -- most correct code I have found out there

# building the model

class S2S_BA_Model(nn.Module):
    def __init__(self, cell_type, input_size, hidden_size, use_cuda):
        super(S2S_BA_Model, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_cuda = use_cuda

        self.cell_type = cell_type

        if self.cell_type not in ['rnn', 'gru', 'lstm']:
            raise ValueError(
                self.cell_type, " is not an appropriate cell type. Please select one of rnn, gru, or lstm.")

        if self.cell_type == 'rnn':
            self.Ecell = nn.RNNCell(self.input_size, self.hidden_size)
            self.Dcell = nn.RNNCell(1+self.hidden_size, self.hidden_size)

        if self.cell_type == 'gru':
            self.Ecell = nn.GRUCell(self.input_size, self.hidden_size)
            self.Dcell = nn.GRUCell(1+self.hidden_size, self.hidden_size)

        if self.cell_type == 'lstm':
            self.Ecell = nn.LSTMCell(self.input_size, self.hidden_size)
            self.Dcell = nn.LSTMCell(1+self.hidden_size, self.hidden_size)

        self.Wattn_energies = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.WValue = nn.Linear(self.hidden_size, 1)
        self.Wout = nn.Linear(1+self.hidden_size*2, self.hidden_size)

        self.v = nn.Parameter(torch.rand(self.hidden_size))
        stdv = 1./math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

        self.init()

    # VERY IMPORTANT INIT PARAMS FUNCTIONS***
    # function to intialize weight parameters

    def init(self):

        if self.cell_type == 'rnn' or self.cell_type == 'gru':

            for p in self.parameters():
                if p.dim() > 1:
                    init.orthogonal_(p.data, gain=1.0)
                if p.dim() == 1:
                    init.constant_(p.data, 0.0)

        elif self.cell_type == 'lstm':

            for p in self.parameters():
                if p.dim() > 1:
                    init.orthogonal_(p.data, gain=1.0)
                if p.dim() == 1:
                    init.constant_(p.data, 0.0)
                    init.constant_(
                        p.data[self.hidden_size:2*self.hidden_size], 1.0)

    def consume(self, x):

        # for rnn and gru
        if self.cell_type == 'rnn' or self.cell_type == 'gru':

            # encoder forward function
            h = torch.zeros(x.shape[0], self.hidden_size)
            encoder_outputs = torch.zeros(
                x.shape[1], x.shape[0], self.hidden_size)

            if self.use_cuda:
                h = h.cuda()
                encoder_outputs = encoder_outputs.cuda()

            # encoder part
            for T in range(x.shape[1]):
                h = self.Ecell(x[:, T, :], h)
                encoder_outputs[T] = h

            pred_Value = self.WValue(h)

        # for lstm
        elif self.cell_type == 'lstm':

            h0 = torch.zeros(x.shape[0], self.hidden_size)
            c0 = torch.zeros(x.shape[0], self.hidden_size)
            encoder_outputs = torch.zeros(
                x.shape[1], x.shape[0], self.hidden_size)

            if self.use_cuda:
                h0 = h0.cuda()
                c0 = c0.cuda()
                encoder_outputs = encoder_outputs.cuda()

            h = (h0, c0)

            for T in range(x.shape[1]):
                h = self.Ecell(x[:, T, :], h)
                encoder_outputs[T] = h[0]

            pred_Value = self.WValue(h[0])

        return pred_Value, h, encoder_outputs

    def predict(self, pred_Value, h, encoder_outputs, target_length):
        # decoder with attention function
        preds = []

        # for rnn and gru
        if self.cell_type == 'rnn' or self.cell_type == 'gru':

            for step in range(target_length):
                h_copies = h.expand(encoder_outputs.shape[0], -1, -1)
                energies = torch.tanh(self.Wattn_energies(
                    torch.cat((h_copies, encoder_outputs), 2)))
                score = torch.sum(self.v * energies, dim=2)
                attn_weights = score.t()
                attn_weights = torch.softmax(attn_weights, dim=1).unsqueeze(1)
                context = attn_weights.bmm(
                    encoder_outputs.transpose(0, 1)).squeeze(1)
                gru_input = torch.cat((pred_Value, context), 1)
                h = self.Dcell(gru_input, h)
                output = self.Wout(torch.cat((pred_Value, h, context), 1))
                pred_Value = self.WValue(output)

                preds.append(pred_Value.unsqueeze(1))

            preds = torch.cat(preds, 1)

        # for lstm
        elif self.cell_type == 'lstm':

            for step in range(target_length):
                h_copies = h[0].expand(encoder_outputs.shape[0], -1, -1)
                energies = torch.tanh(self.Wattn_energies(
                    torch.cat((h_copies, encoder_outputs), 2)))
                score = torch.sum(self.v * energies, dim=2)
                attn_weights = score.t()
                attn_weights = torch.softmax(attn_weights, dim=1).unsqueeze(1)
                context = attn_weights.bmm(
                    encoder_outputs.transpose(0, 1)).squeeze(1)
                gru_input = torch.cat((pred_Value, context), 1)
                h = self.Dcell(gru_input, h)
                output = self.Wout(torch.cat((pred_Value, h[0], context), 1))
                pred_Value = self.WValue(output)

                preds.append(pred_Value.unsqueeze(1))

            preds = torch.cat(preds, 1)

        return preds


# 3

# Luong Attention module

class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method

        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(
                self.method, " is not an appropriate attention method, please select one of dot, general, or concat.")

        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)

        if self.method == 'concat':
            self.attn = nn.Linear(2*self.hidden_size, self.hidden_size)
            self.v = nn.Parameter(torch.rand(self.hidden_size))
            stdv = 1./math.sqrt(self.v.size(0))
            self.v.data.normal_(mean=0, std=stdv)

    def dot_score(self, hidden, encoder_output):
        attn_energies = torch.sum(hidden*encoder_output, dim=2)
        return attn_energies

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        attn_energies = torch.sum(hidden*energy, dim=2)
        return attn_energies

    def concat_score(self, hidden, encoder_output):
        energy = torch.tanh(self.attn(torch.cat((hidden.expand(encoder_output.shape[0], -1, -1),
                                                encoder_output), 2)))
        return torch.sum(self.v * energy, dim=2)

    # calculate the attention weights (energies) based on the given method
    def forward(self, hidden, encoder_outputs):

        if self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        elif self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)

        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)

        attn_energies = attn_energies.t()
        attn_weights = torch.softmax(attn_energies, dim=1).unsqueeze(1)

        return attn_weights


#########################################################################################

#  building the S2S LA model

class S2S_LA_Model(nn.Module):
    def __init__(self, cell_type, attn_method, input_size, hidden_size, use_cuda):
        super(S2S_LA_Model, self).__init__()

        self.cell_type = cell_type
        self.attn_method = attn_method
        self.input_size = input_size
        self.hidden_size = hidden_size

        if self.cell_type == 'rnn':
            self.Ecell = nn.RNNCell(self.input_size, self.hidden_size)
            self.Dcell = nn.RNNCell(1, self.hidden_size)

        if self.cell_type == 'gru':
            self.Ecell = nn.GRUCell(self.input_size, self.hidden_size)
            self.Dcell = nn.GRUCell(1, self.hidden_size)

        if self.cell_type == 'lstm':
            self.Ecell = nn.LSTMCell(self.input_size, self.hidden_size)
            self.Dcell = nn.LSTMCell(1, self.hidden_size)

        self.lin_Value = nn.Linear(hidden_size, 1)
        self.lin_concat = nn.Linear(hidden_size*2, hidden_size)

        self.attn = Attn(self.attn_method, self.hidden_size)

        self.use_cuda = use_cuda

        self.init()

    # VERY IMPORTANT INIT PARAMS FUNCTIONS***
    # function to intialize weight parameters

    def init(self):

        if self.cell_type == 'rnn' or self.cell_type == 'gru':

            for p in self.parameters():
                if p.dim() > 1:
                    init.orthogonal_(p.data, gain=1.0)
                if p.dim() == 1:
                    init.constant_(p.data, 0.0)

        elif self.cell_type == 'lstm':

            for p in self.parameters():
                if p.dim() > 1:
                    init.orthogonal_(p.data, gain=1.0)
                if p.dim() == 1:
                    init.constant_(p.data, 0.0)
                    init.constant_(
                        p.data[self.hidden_size:2*self.hidden_size], 1.0)

    def consume(self, x):

        if self.cell_type == 'rnn' or self.cell_type == 'gru':

            # encoder forward function
            h = torch.zeros(x.shape[0], self.hidden_size)
            encoder_outputs = torch.zeros(
                x.shape[1], x.shape[0], self.hidden_size)

            if self.use_cuda:
                h = h.cuda()
                encoder_outputs = encoder_outputs.cuda()

            # encoder part
            for T in range(x.shape[1]):
                h = self.Ecell(x[:, T, :], h)
                encoder_outputs[T] = h

            pred_Value = self.lin_Value(h)

        elif self.cell_type == 'lstm':

            h0 = torch.zeros(x.shape[0], self.hidden_size)
            c0 = torch.zeros(x.shape[0], self.hidden_size)
            encoder_outputs = torch.zeros(
                x.shape[1], x.shape[0], self.hidden_size)

            if self.use_cuda:
                h0 = h0.cuda()
                c0 = c0.cuda()
                encoder_outputs = encoder_outputs.cuda()

            h = (h0, c0)

            for T in range(x.shape[1]):
                h = self.Ecell(x[:, T, :], h)
                encoder_outputs[T] = h[0]

            pred_Value = self.lin_Value(h[0])

        return pred_Value, h, encoder_outputs

    def predict(self, pred_Value, h, encoder_outputs, target_length):
        # decoder with attention function
        preds = []

        if self.cell_type == 'rnn' or self.cell_type == 'gru':

            for step in range(target_length):
                h = self.Dcell(pred_Value, h)
                attn_weights = self.attn(h, encoder_outputs)
                context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
                context = context.squeeze(1)
                concat_input = torch.cat((h, context), 1)
                concat_output = torch.tanh(self.lin_concat(concat_input))
                pred_Value = self.lin_Value(concat_output)

                preds.append(pred_Value.unsqueeze(1))

            preds = torch.cat(preds, 1)

        elif self.cell_type == 'lstm':

            for step in range(target_length):
                h = self.Dcell(pred_Value, h)
                attn_weights = self.attn(h[0], encoder_outputs)
                context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
                context = context.squeeze(1)
                concat_input = torch.cat((h[0], context), 1)
                concat_output = torch.tanh(self.lin_concat(concat_input))
                pred_Value = self.lin_Value(concat_output)

                preds.append(pred_Value.unsqueeze(1))

            preds = torch.cat(preds, 1)

        return preds


#################################################################################################################################################

# main function

def main(dataset, seed, cuda, cell_type, attention_model, la_method, window_source_size,
         window_target_size, epochs, batch_size, hs, save_model, loss_fn):

    t0 = time.time()

    # seed == given seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset['epochdate'] = dataset['Date_Time'].map(pd.Timestamp.timestamp).astype(np.float32)

	# drop other date columns
    dataset = dataset.drop(columns=['Date', 'Date_Time'])

    Value_actual = dataset['Value']

    mu_Value = dataset['Value'].mean()
    std_Value = dataset['Value'].std()

    dataset = dataset.values

    # 0 mean and unit var
    print("Transforming data to 0 mean and unit var")
    MU = dataset.mean(0)  # 0 means take the mean of the column
    dataset = dataset - MU

    STD = dataset.std(0)  # same with std here
    dataset = dataset / STD

    print("Generating training and test data...")
    WINDOW_SOURCE_SIZE = window_source_size
    WINDOW_TARGET_SIZE = window_target_size

    # getting actual Value vector, aligning with predicted values vector. Aka remove first window_source_size and remaining
    Value_actual = Value_actual.values
    Value_actual = Value_actual[int(dataset.shape[0]*0.80):]
    Value_actual = Value_actual[WINDOW_SOURCE_SIZE:]

    # 80% of the data will be train
    # 20% is test
    train_source = dataset[:int(dataset.shape[0]*0.80)]
    test_source = dataset[int(dataset.shape[0]*0.80):]

    # if N = data.shape[0] - (WINDOW_SOURCE_SIZE+WINDOW_TARGET_SIZE)
    # then you will be sliding over every one

    def generate_windows(data):
        x_train = []
        y_Value_train = []

        x_test = []
        y_Value_test = []

        # for training data
        idxs = np.random.choice(train_source.shape[0]-(WINDOW_SOURCE_SIZE+WINDOW_TARGET_SIZE),
                                train_source.shape[0]-(WINDOW_SOURCE_SIZE+WINDOW_TARGET_SIZE), replace=False)

        for idx in idxs:
            x_train.append(train_source[idx:idx+WINDOW_SOURCE_SIZE].reshape(
                (1, WINDOW_SOURCE_SIZE, train_source.shape[1])))
            y_Value_train.append(train_source[idx+WINDOW_SOURCE_SIZE:idx+WINDOW_SOURCE_SIZE +
                                 WINDOW_TARGET_SIZE, -1].reshape((1, WINDOW_TARGET_SIZE, 1)))

        # make them arrays and not lists
        x_train = np.concatenate(x_train, axis=0)
        y_Value_train = np.concatenate(y_Value_train, axis=0)

        # for testing data
        idxs = np.arange(0, len(test_source)-(WINDOW_SOURCE_SIZE +
                         WINDOW_TARGET_SIZE), WINDOW_TARGET_SIZE)

        for idx in idxs:
            x_test.append(test_source[idx:idx+WINDOW_SOURCE_SIZE].reshape(
                (1, WINDOW_SOURCE_SIZE, test_source.shape[1])))
            y_Value_test.append(test_source[idx+WINDOW_SOURCE_SIZE:idx+WINDOW_SOURCE_SIZE +
                                WINDOW_TARGET_SIZE, -1].reshape((1, WINDOW_TARGET_SIZE, 1)))

        # make them arrays and not lists
        x_test = np.concatenate(x_test, axis=0)
        y_Value_test = np.concatenate(y_Value_test, axis=0)

        return x_train, y_Value_train, x_test, y_Value_test

    X_train, Y_train_Value, X_test, Y_test_Value = generate_windows(dataset)
    print("Created {} train samples and {} test samples".format(
        X_train.shape[0], X_test.shape[0]))

    idxs = np.arange(0, len(test_source)-(WINDOW_SOURCE_SIZE +
                     WINDOW_TARGET_SIZE), WINDOW_TARGET_SIZE)
    remainder = len(test_source) - \
        (idxs[-1] + WINDOW_SOURCE_SIZE+WINDOW_TARGET_SIZE)

    Value_actual = Value_actual[:-remainder]

    #################################################################################################################################################

    # call the model

    #print("Creating model...")
    INPUT_SIZE = X_train.shape[-1]
    HIDDEN_SIZE = hs
    CELL_TYPE = cell_type
    LA_METHOD = la_method

    # call the respective model
    if attention_model == 'none':
        model = S2S_Model(CELL_TYPE, INPUT_SIZE, HIDDEN_SIZE, use_cuda=cuda)

    elif attention_model == 'BA':
        model = S2S_BA_Model(CELL_TYPE, INPUT_SIZE, HIDDEN_SIZE, use_cuda=cuda)

    elif attention_model == 'LA':
        model = S2S_LA_Model(CELL_TYPE, LA_METHOD,
                             INPUT_SIZE, HIDDEN_SIZE, use_cuda=cuda)

    if cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model.cuda()

    print("MODEL ARCHITECTURE IS: ")
    print(model)

    print("\nModel parameters are on cuda: {}".format(
        next(model.parameters()).is_cuda))

    opt = optim.Adam(model.parameters(), lr=1e-3)
    # loss_fn = nn.MSELoss(reduction='sum')
    EPOCHES = epochs
    BATCH_SIZE = batch_size

    print("\nStarting training...")

    train_loss = []
    test_loss = []

    for epoch in range(EPOCHES):

        t_one_epoch = time.time()

        print("Epoch {}".format(epoch+1))

        total_Value_loss = 0

        for b_idx in range(0, X_train.shape[0], BATCH_SIZE):

            x = torch.from_numpy(X_train[b_idx:b_idx+BATCH_SIZE]).float()
            y_Value = torch.from_numpy(
                Y_train_Value[b_idx:b_idx+BATCH_SIZE]).float()

            if cuda:
                x = x.cuda()
                y_Value = y_Value.cuda()

            # encoder forward, for respective models (with and without attention)
            if attention_model == 'none':
                pred_Value, h = model.consume(x)

            elif attention_model == 'BA':
                pred_Value, h, encoder_outputs = model.consume(x)

            elif attention_model == 'LA':
                pred_Value, h, encoder_outputs = model.consume(x)

            # decoder forward, for respective models
            if attention_model == 'none':
                preds = model.predict(pred_Value, h, WINDOW_TARGET_SIZE)

            elif attention_model == 'BA':
                preds = model.predict(
                    pred_Value, h, encoder_outputs, WINDOW_TARGET_SIZE)

            elif attention_model == 'LA':
                preds = model.predict(
                    pred_Value, h, encoder_outputs, WINDOW_TARGET_SIZE)

            # compute lose
            loss_Value = loss_fn(preds, y_Value)

            # backprop and update
            opt.zero_grad()

            loss_Value.backward()

            opt.step()

            total_Value_loss += loss_Value.item()

        train_loss.append(total_Value_loss)

        print("\tTRAINING: {} total train Value loss.\n".format(total_Value_loss))

        #################################################################################################################################################
        # TESTING

        y_Value = None
        pred_Value = None
        preds = None

        total_Value_loss = 0

        all_preds = []

        for b_idx in range(0, X_test.shape[0], BATCH_SIZE):
            with torch.no_grad():

                x = torch.from_numpy(X_test[b_idx:b_idx+BATCH_SIZE])
                y_Value = torch.from_numpy(
                    Y_test_Value[b_idx:b_idx+BATCH_SIZE])

                if cuda:
                    x = x.cuda()
                    y_Value = y_Value.cuda()

                # encoder forward, for respective models
                if attention_model == 'none':
                    pred_Value, h = model.consume(x)

                elif attention_model == 'BA':
                    pred_Value, h, encoder_outputs = model.consume(x)

                elif attention_model == 'LA':
                    pred_Value, h, encoder_outputs = model.consume(x)

                # decoder forward, for respective models
                if attention_model == 'none':
                    preds = model.predict(pred_Value, h, WINDOW_TARGET_SIZE)

                elif attention_model == 'BA':
                    preds = model.predict(
                        pred_Value, h, encoder_outputs, WINDOW_TARGET_SIZE)

                elif attention_model == 'LA':
                    preds = model.predict(
                        pred_Value, h, encoder_outputs, WINDOW_TARGET_SIZE)

                # compute loss
                loss_Value = loss_fn(preds, y_Value)

                total_Value_loss += loss_Value.item()

                if (epoch == epochs-1):
                    all_preds.append(preds)

        test_loss.append(total_Value_loss)

        print("\tTESTING: {} total test Value loss".format(total_Value_loss))

        print("\tTESTING:\n")
        print("\tSample of prediction:")
        print("\t\t TARGET: {}".format(
            y_Value[-1].cpu().detach().numpy().flatten()))
        print("\t\t   PRED: {}\n\n".format(
            preds[-1].cpu().detach().numpy().flatten()))

        y_last_Value = y_Value[-1].cpu().detach().numpy().flatten()
        pred_last_Value = preds[-1].cpu().detach().numpy().flatten()

        t2_one_epoch = time.time()

        time_one_epoch = t2_one_epoch - t_one_epoch

        print("TIME OF ONE EPOCH: {} seconds and {} minutes".format(
            time_one_epoch, time_one_epoch/60.0))

    ####################################################################################################
    # SAVING MODEL

    if save_model:
        torch.save(model.state_dict(), "MODEL_w:__seed={}_cell_type={}_attention_model={}_la_method={}_T={}_N={}_bs={}_hs={}".format(
            seed, cell_type, attention_model, la_method,
            window_source_size, window_target_size, batch_size, hs))

    #################################################################################################################################################

    # PLOTTING

    # for plotting and accuracy
    preds = torch.cat(all_preds, 0)
    preds = preds.cpu().detach().numpy().flatten()

    actual = Y_test_Value.flatten()

    # for loss plotting
    train_loss_array = np.asarray(train_loss)
    test_loss_array = np.asarray(test_loss)

    len_loss = np.arange(len(train_loss_array))

    # unnormalizing 1
    preds_unnorm = (preds*std_Value) + mu_Value

    # using the actual Value from top of script here
    mae3 = (sum(abs(Value_actual - preds_unnorm)))/(len(Value_actual))
    mape3 = (sum(abs((Value_actual - preds_unnorm)/Value_actual))) / \
        (len(Value_actual))
    smape3 = (sum(abs(Value_actual - preds_unnorm)/(abs(Value_actual)+abs(peds_innorm))))) / \
        (len(Value_actual))

    # for std
    mape_s = (abs((Value_actual - preds_unnorm)/Value_actual))
    s = mape_s.std()

    mae_s = abs(Value_actual - preds_unnorm)
    s2 = mae_s.std()

    print("\n\tACTUAL ACC. RESULTS: MAE, MAPE, SMAPE: {}, {}%, and {}%".format(mae3, mape3*100.0, smape3*100.0))

    # plotting
    # plt.figure(1)
    # plt.plot(np.arange(len(preds)), preds, 'b', label='Predicted')
    # plt.plot(np.arange(len(actual)), actual, 'g', label='Actual')
    # plt.title("Predicted vs Actual, {} timesteps to {} timesteps".format(
    #     window_source_size, window_target_size))
    # plt.xlabel("Time in 15 minute increments")
    # plt.ylabel("Value (normalized)")
    # plt.legend(loc='lower left')

    # plt.figure(2)
    # plt.plot(np.arange(len(actual)), actual, 'g', label='Actual')
    # plt.plot(np.arange(len(preds)), preds, 'b', label='Predicted')
    # plt.title("Predicted vs Actual, {} timesteps to {} timesteps".format(
    #     window_source_size, window_target_size))
    # plt.xlabel("Time in 15 minute increments")
    # plt.ylabel("Value (normalized)")
    # plt.legend(loc='lower left')

    # plt.figure(3)
    # plt.plot(np.arange(len(y_last_Value)), y_last_Value, 'g', label='Actual')
    # plt.plot(np.arange(len(pred_last_Value)),
    #          pred_last_Value, 'b', label='Predicted')
    # plt.title("Predicted vs Actual last test example, {} timesteps to {} timesteps".format(
    #     window_source_size, window_target_size))
    # plt.xlabel("Time in 15 minute increments")
    # plt.ylabel("Value (normalized)")
    # plt.legend(loc='lower left')

    plt.figure(4)
    plt.plot(np.arange(len(Value_actual[-4*24*30:])),
             Value_actual[-4*24*30:], 'g', label='Actual')
    plt.plot(np.arange(len(preds_unnorm[-4*24*30:])),
             preds_unnorm[-4*24*30:], 'b', label='Predicted')
    plt.title("Predicted vs Actual: Case 2, Zoom last 30 days".format(
        window_source_size, window_target_size))
    plt.xlabel("Time in 15 minute increments")
    plt.ylabel("Value (kW)")
    plt.legend(loc='lower left')

    plt.figure(5)
    plt.plot(len_loss, train_loss_array, 'k')
    plt.title("Train loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.figure(6)
    plt.plot(len_loss, test_loss_array, 'r')
    plt.title("Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.figure(7)
    plt.plot(len_loss, test_loss_array, 'r')
    ylim = plt.gca().get_ylim()
    plt.plot(len_loss, train_loss_array, 'k')
    plt.gca().set_ylim(ylim)
    plt.title("Train and Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # total time of run
    t1 = time.time()
    total = t1-t0
    print("\nTIME ELAPSED: {} seconds OR {} minutes".format(total, total/60.0))

    print("\nEnd of run")

    plt.show()

    for_plotting = [Value_actual, preds_unnorm, y_last_Value, pred_last_Value]

    return s, s2, mape_s, mae_s, mae3, mape3, smape3, total/60.0, train_loss, test_loss, for_plotting


############################################################################################################################
# for one specific model at a time:
if __name__ == "__main__":

    print("Loading dataset...")
    dataset = pd.read_csv(
		'data/run-of-river_production_load.csv', 
		dtype=np.float32,
		converters={
			"Date_Time": pd.to_datetime,
			"Date": pd.to_datetime
    })

    s, s2, mape_s, mae_s, mae, mape, smape, total_mins, train_loss, test_loss, for_plotting = main(
        dataset=dataset,
		seed=0, # for reproducability
		cuda=False, # change to True if available on your platform
		cell_type='gru', attention_model='BA', la_method='none', # model architecture
		window_source_size=96 * 4,
		window_target_size=24,
		epochs=10, batch_size=128, hs=48, # overall training parameters
		save_model=False,
        loss_fn=nn.MSELoss(reduction='sum')
		)
