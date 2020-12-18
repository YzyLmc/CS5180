#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 11:24:42 2020

@author: ziyi
"""


from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

from copy import deepcopy
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
#import calculatebleu
#from calculatebleu import BLEU
from nltk.translate.bleu_score import sentence_bleu as BLEU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device('cpu')
SOS_token = 0
EOS_token = 1
padding = 2


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2:"<pad>"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
            
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    return len(p[0].split(' ')) > 2 and \
        len(p[1].split(' ')) > 2 and \
        len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
data_num = len(pairs)
pairs_train = pairs[:int(0.8*data_num)]
pairs_test = pairs[int(0.8*data_num):]
print(random.choice(pairs))

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
  
class DecoderRNN2(nn.Module):       ##same encoder but different decoder, output_size = 1
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN2, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)
        

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0])
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)  
   
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    
    decoded_tokens = torch.zeros(target_length)
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            #print(decoder_output)
            decoded_tokens[di] = topi.squeeze().detach().clone()
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden  = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            #print(decoder_output)
            decoded_tokens[di] = topi.squeeze().detach().clone()
            decoder_input = topi.squeeze().detach()  # detach from history as input
            #try:
            loss += criterion(decoder_output, target_tensor[di])
            #except:
            #    loss += criterion(decoder_output,2)
            if decoder_input.item() == EOS_token:
                decoded_tokens = decoded_tokens[:di + 1]
                break
    #print(loss)
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()
    #print(target_tensor.view(1,-1).int().tolist(),decoded_tokens.int().tolist())
    #return loss.item() / target_length
    return BLEU(target_tensor.view(1,-1).int().tolist(),decoded_tokens.int().tolist(),weights=(1/3,1/3,1/3))
import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def pretrain1itr(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs_train))
                      for i in range(n_iters)]
    #print(training_pairs)
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    #showPlot(plot_losses)
    #print(plot_losses)
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    #loc = ticker.MultipleLocator(base=0.2)
    #ax.yaxis.set_major_locator(loc)
    ax.plot(plot_losses)
    ax.set_title('BLEU score SL')
    plt.show()
    
import matplotlib.pyplot as plt
#plt.switch_backend('GTKAgg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()
    fig.show()
    
        #%%
    
def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            
            topv, topi = decoder_output.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])
            decoder_input = topi.squeeze().detach()

        return decoded_words
    
def evaluateN(encoder, decoder, sentence, ref, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()
        target_tensor = tensorFromSentence(output_lang,ref)
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            
            topv, topi = decoder_output.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append(1)
                break
            else:
                decoded_words.append(topi.item())
            decoder_input = topi.squeeze().detach()
        #print(decoded_words, target_tensor.view(1,1,-1).int().tolist())
        return BLEU(target_tensor.view(1,-1).int().tolist(),decoded_words,weights=(1/3,1/3,1/3))
    
def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs_test)
        print('>', pair[0])
        print('=', pair[1])
        output_words = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
        
def evaluateNRandomly(encoder, decoder, n=10):
    bleuLs = []
    for i in range(n):
        pair = random.choice(pairs_test)
        bleui = evaluateN(encoder, decoder, pair[0], pair[1])
        bleuLs.append(bleui)
    
    print('Avg BLEU score', np.mean(bleuLs))
#%%
hidden_size = 64
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder1 = DecoderRNN(hidden_size, output_lang.n_words).to(device)
#%%
encoder2 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder2 = DecoderRNN2(hidden_size, output_lang.n_words).to(device)
#%% pretrain actor network
pretrain1itr(encoder1, decoder1, 5000, print_every= 500)

evaluateRandomly(encoder1, decoder1)
#%% 
def pretrain2(input_tensor, target_tensor, encoder1, decoder1,encoder2, decoder2, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    with torch.no_grad():
        #input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        target_length = target_tensor.size()[0]
        encoder_hidden = encoder1.initHidden()
        #target_tensor = tensorFromSentence(output_lang,ref)
        encoder_outputs = torch.zeros(max_length, encoder1.hidden_size, device=device)
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder1(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoded_tokens = []

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder1(
                decoder_input, decoder_hidden)
            #print(decoder_input)
            topv, topi = decoder_output.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append(1)
                decoded_tokens.append(torch.tensor([[EOS_token]], device = device))
                break
            else:
                decoded_words.append(topi.item())
                decoded_tokens.append(topi)
            decoder_input = topi.squeeze().detach()
    
    returnLs = torch.zeros(len(decoded_words), device=device) 
    #print(decoded_words,decoded_tokens)
    G = 0
    for j in range(len(decoded_words)):
                    
        G = G + BLEU(target_tensor.view(1,-1).int().tolist(),decoded_words[:len(decoded_words) - j],weights=(1/3,1/3,1/3)) - BLEU(target_tensor.view(1,-1).int().tolist(),decoded_words[:len(decoded_words)-j-1],weights=(1/3,1/3,1/3))     
        
        returnLs[len(decoded_words)-j-1] = G
    #print(returnLs)
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    for ei in range(target_length):
        
        encoder_output, encoder_hidden = encoder2(
            target_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]
        
    decoder_input = torch.tensor([[SOS_token]],device = device)

    decoder_hidden = encoder_hidden
    loss = 0
    for di in range(len(decoded_words)):
        #print(decoder_input)
        #print(loss)
        decoder_output, decoder_hidden  = decoder2(
            decoder_input, decoder_hidden)
        #print(decoder_output, returnLs[di])
        #print(decoder_output-returnLs[di])
        loss += criterion(decoder_output,returnLs[di])
        #loss += (decoder_output-returnLs[di])*(decoder_output-returnLs[di])
        decoder_input = decoded_tokens[di].squeeze().detach()
    #print(loss)
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.item() / target_length
    
def pretrain2itr(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.0001):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs_train))
                      for i in range(n_iters)]
    #print(training_pairs)
    criterion = nn.MSELoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = pretrain2(input_tensor, target_tensor, encoder1,
                     decoder1,encoder2,decoder2, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    #showPlot(plot_losses)
    #print(plot_losses)
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    #loc = ticker.MultipleLocator(base=0.2)
    #ax.yaxis.set_major_locator(loc)
    ax.plot(plot_losses)
    ax.set_title('Critic pretrain')
    plt.show()
    
import matplotlib.pyplot as plt
#plt.switch_backend('GTKAgg')
import matplotlib.ticker as ticker
import numpy as np

pretrain2itr(encoder2, decoder2, 1000, print_every= 100)
#%% delayed actor and target critic
encoder1d = deepcopy(encoder1)
decoder1d = deepcopy(decoder1)
encoder2t = deepcopy(encoder2)
decoder2t = deepcopy(decoder2)
#%%algo 1, actor-critic
def trainAC(input_tensor, target_tensor, encoder1, decoder1,encoder2, decoder2, encoder1d, decoder1d, encoder2t, decoder2t, encoder1_optimizer, decoder1_optimizer,encoder2_optimizer, decoder2_optimizer,criterion,alpha ,max_length=MAX_LENGTH):
    encoder_hidden = encoder1.initHidden()
    
    encoder1_optimizer.zero_grad()
    decoder1_optimizer.zero_grad()
    encoder2_optimizer.zero_grad()
    decoder2_optimizer.zero_grad()
    
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder1d.hidden_size, device=device)

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder1d(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    output_soft = []#torch.zeros(target_length)
    decoded_tokens = torch.zeros(max_length,dtype = torch.long, device = device)
    softmax = nn.Softmax(dim=1)
    
    lossSL = 0
    for di in range(max_length):
        decoder_output, decoder_hidden  = decoder1d(
            decoder_input, decoder_hidden)
        soft = softmax(decoder_output)
        topi = soft.multinomial(num_samples=1, replacement=True)
        topv = soft[0,topi]
        output_soft.append(topv)
        decoded_tokens[di] = topi.squeeze().detach()
        decoder_input = topi.squeeze().detach()  # detach from history as input
        try:
            lossSL += criterion(decoder_output, target_tensor[di])
        except:
            lossSL += criterion(decoder_output,torch.tensor([1],device = device))
        #lossSL += criterion(decoder_output, target_tensor[di])
        if decoder_input.item() == EOS_token:
            decoded_tokens = decoded_tokens[:di + 1]
            break
    lossRL = 0  
    for i in range(len(output_soft)-1):
        
        G = 0
        for j in range(len(output_soft)-i-1,len(output_soft)):
            if j > 1:
                
                #print(target_tensor.view(1,-1).int().tolist(),decoded_tokens[:j].int().tolist())
                #print(BLEU(decoded_tokens[:j].view(1,-1).int().tolist(),target_tensor.view(1,1,-1).int().tolist()) - BLEU(decoded_tokens[:j-1].view(1,-1).int().tolist(),target_tensor.view(1,1,-1).int().tolist()))
                G = G + BLEU(target_tensor.view(1,-1).int().tolist(),decoded_tokens[:j].int().tolist(),weights=(1/3,1/3,1/3)) - BLEU(target_tensor.view(1,-1).int().tolist(),decoded_tokens[:j-1].int().tolist(),weights=(1/3,1/3,1/3))
            else:
                G =  G + BLEU(target_tensor.view(1,-1).int().tolist(),decoded_tokens[:j].int().tolist(),weights=(1/3,1/3,1/3))
            
      
        lossRL += - G * torch.log(output_soft[len(output_soft)-i-1])   
     
        
    loss = (1 - alpha) *lossRL + alpha *lossSL
    #print(lossRL, loss)
    loss.backward(retain_graph=True)
    
    encoder1_optimizer.step()
    decoder1_optimizer.step()
    
    returnLs = torch.zeros(len(decoded_tokens), device=device) 
    #print(decoded_words,decoded_tokens)
    G = 0
    for j in range(len(decoded_tokens)):
                    
        G = G + BLEU(target_tensor.view(1,-1).int().tolist(),decoded_tokens[:len(decoded_tokens) - j],weights=(1/3,1/3,1/3)) - BLEU(target_tensor.view(1,-1).int().tolist(),decoded_tokens[:len(decoded_tokens)-j-1],weights=(1/3,1/3,1/3))     
        
        returnLs[len(decoded_tokens)-j-1] = G
        
        
    for ei in range(target_length):
        
        encoder_output, encoder_hidden = encoder2(
            target_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]
        
    decoder_input = torch.tensor([[SOS_token]],device = device)

    decoder_hidden = encoder_hidden
    loss = 0
    mle = nn.MSELoss()
    for di in range(len(decoded_tokens)):
        
        decoder_output, decoder_hidden  = decoder2(
            decoder_input, decoder_hidden)
        loss += mle(decoder_output,returnLs[di])
        #loss += (decoder_output-returnLs[di])*(decoder_output-returnLs[di])
        #print(decoded_tokens[di])
        decoder_input = decoded_tokens[di].squeeze().detach()
    #print(loss)
    loss.backward()

    encoder2_optimizer.step()
    decoder2_optimizer.step() 
    
    #e1param = list(encoder1.parameters())
    #for param in encoder1d.parameters(): param = param*0.95 + 
    encoder1d = deepcopy(encoder1)
    decoder1d = deepcopy(decoder1)
    encoder2t = deepcopy( encoder2)
    decoder2t = deepcopy( decoder2)
    
    return BLEU(target_tensor.view(1,-1).int().tolist(),decoded_tokens.int().tolist(),weights=(1/3,1/3,1/3))

def trainItersAC(encoder, decoder, n_iters, print_every=1000, plot_every=200, learning_rate=0.001):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder1_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder1_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    encoder2_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder2_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs_train))
                      for i in range(n_iters)]
    #print(training_pairs)

    criterion = nn.NLLLoss()
    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = trainAC(input_tensor, target_tensor,encoder1, decoder1,encoder2, decoder2, encoder1d, decoder1d, encoder2t, decoder2t, encoder1_optimizer, decoder1_optimizer,encoder2_optimizer, decoder2_optimizer, criterion,alpha = 0.3)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    #showPlot(plot_losses)
    #print(plot_losses)
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    #loc = ticker.MultipleLocator(base=0.2)
    #ax.yaxis.set_major_locator(loc)
    ax.plot(plot_losses)
    ax.set_title('BLEU score Actor-Critic')
    plt.show()
    
#with torch.autograd.set_detect_anomaly(True):
trainItersAC(encoder1, decoder1, 50000, print_every=5000)
#%%
evaluateRandomly(encoder1d, decoder1d)




