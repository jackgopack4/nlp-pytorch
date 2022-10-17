# models.py

import numpy as np
import collections
import torch
import torch.nn as nn
from torch import optim
import random

#####################
# MODELS FOR PART 1 #
#####################
class ConsonantDataset(torch.utils.data.Dataset):
    def __init__(self,consonant_examples, vowel_examples, indexer):
        self.indexer = indexer
        self.text = []
        self.labels = []
        for c in consonant_examples:
            x = self.form_input(c)
            y = torch.tensor([0],dtype=torch.float)
            self.text.append(x)
            self.labels.append(y)
        for v in vowel_examples:
            x = self.form_input(v)
            y = torch.tensor([1],dtype=torch.float)
            self.text.append(x)
            self.labels.append(y)
        #templist = list(zip(self.text,self.labels))
        #random.shuffle(templist)
        #self.text, self.labels = zip ( * templist)

    def __len__(self):
        return len(self.text)

    def __getitem__(self,idx):
        return self.text[idx], self.labels[idx]

    def form_input(self,phrase) -> torch.Tensor:
        charlist = list(phrase)
        chartensor = torch.zeros(20,dtype=torch.long)
        i:int = 0
        for c in charlist:
            chartensor[i]=self.indexer.index_of(c)
            i+=1
        return chartensor
class ConsonantVowelClassifier(object):
    def predict(self, context):
        """
        :param context:
        :return: 1 if vowel, 0 if consonant
        """
        raise Exception("Only implemented in subclasses")


class FrequencyBasedClassifier(ConsonantVowelClassifier):
    """
    Classifier based on the last letter before the space. If it has occurred with more consonants than vowels,
    classify as consonant, otherwise as vowel.
    """
    def __init__(self, consonant_counts, vowel_counts):
        self.consonant_counts = consonant_counts
        self.vowel_counts = vowel_counts

    def predict(self, context):
        # Look two back to find the letter before the space
        if self.consonant_counts[context[-1]] > self.vowel_counts[context[-1]]:
            return 0
        else:
            return 1


class RNNClassifier(ConsonantVowelClassifier, nn.Module):
    def __init__(self, dict_size, classify_size, input_size, hidden_size,num_layers, dropout, vocab_index,rnn_type='lstm'):
        super(RNNClassifier, self).__init__()
        torch.manual_seed(420)
        random.seed(420)
        self.classify_size = classify_size
        self.num_layers = num_layers
        self.vocab_index = vocab_index
        self.word_embedding = nn.Embedding(dict_size, input_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, num_layers=num_layers, dropout=dropout,batch_first=True)
        self.init_weight()
        self.linear = nn.Linear(hidden_size, classify_size)
        self.dd = nn.Dropout(p=0.1)
        self.sm = nn.LogSoftmax(dim=0)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def form_input(self,phrase) -> torch.Tensor:
        charlist = list(phrase)
        chartensor = torch.zeros(20,dtype=torch.long)
        i:int = 0
        for c in charlist:
            chartensor[i]=self.vocab_index.index_of(c)
            i+=1
        return chartensor.unsqueeze(0)

    def predict(self, input):
        self.debug = True
        self.eval()
        x = self.form_input(input)
        loss_prob = self.forward(x).squeeze(0).log_softmax(dim=0)
        prediction = torch.argmax(loss_prob)
        #print(prediction)
        #print(f'input: {input} log prob: {loss_prob},\t prediction: {prediction}')
        return prediction

    def init_weight(self):
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
        nn.init.uniform_(self.rnn.bias_hh_l0)
        nn.init.uniform_(self.rnn.bias_ih_l0)

    def forward(self, input):
        # adapted from https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_lstm_neuralnetwork/#step-3-create-model-class
        h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).requires_grad_()
        embedded_input = self.word_embedding(input)
        o,(h,c) = self.rnn(embedded_input,(h0.detach(),c0.detach()))
        lin = self.linear(o[:,-1,:])
        #print(lin)
        #probs = self.sm(lin)
        #print(probs)
        #print(probs.size())
        return lin


def train_frequency_based_classifier(cons_exs, vowel_exs):
    consonant_counts = collections.Counter()
    vowel_counts = collections.Counter()
    for ex in cons_exs:
        consonant_counts[ex[-1]] += 1
    for ex in vowel_exs:
        vowel_counts[ex[-1]] += 1
    return FrequencyBasedClassifier(consonant_counts, vowel_counts)


def train_rnn_classifier(args, train_cons_exs, train_vowel_exs, dev_cons_exs, dev_vowel_exs, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_cons_exs: list of strings followed by consonants
    :param train_vowel_exs: list of strings followed by vowels
    :param dev_cons_exs: list of strings followed by consonants
    :param dev_vowel_exs: list of strings followed by vowels
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNClassifier instance trained on the given data
    """
    rnn_module = RNNClassifier(dict_size=27,classify_size=2, input_size=8, hidden_size=20,num_layers=1, dropout=0.0,vocab_index=vocab_index)
    initial_learning_rate = 0.05
    optimizer = optim.SGD(rnn_module.parameters(), lr = initial_learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 50,factor=0.5)
    loss_func = nn.CrossEntropyLoss()
    num_epochs = 500
    random.shuffle(train_cons_exs)
    random.shuffle(train_vowel_exs)
    train_dataset = ConsonantDataset(train_cons_exs,train_vowel_exs,vocab_index)
    test_dataset = ConsonantDataset(dev_cons_exs,dev_vowel_exs,vocab_index)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=200, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=40, shuffle=True)
    for epoch in range(num_epochs):
        train_loss = 0.0
        test_loss = 0.0
        rnn_module.train()
        for x,y in train_dataloader:
            #rnn_module.zero_grad()
            optimizer.zero_grad()
            loss_prob = rnn_module(x)
            #print(loss_prob.size())
            loss = loss_func(loss_prob, y.squeeze(1).long())
            loss.backward()
            optimizer.step()
            train_loss+= loss.item()*x.size(0)
        rnn_module.eval()
        scheduler.step(train_loss)
        curr_lr = optimizer.param_groups[0]['lr']
        for x,y in test_dataloader:
            loss_prob = rnn_module(x)
            loss = loss_func(loss_prob,y.squeeze(1).long())
            test_loss+=loss.item()*x.size(0)
        #Validation Loss: {test_loss/len(test_dataloader.dataset)}\t \
        scheduler.step(test_loss)
        #print(f'Epoch {epoch}\t \
        #    Training Loss: {train_loss}\t \
        #    Test Loss: {test_loss}\t \
        #    LR:{curr_lr}')
    return rnn_module



#####################
# MODELS FOR PART 2 #
#####################


class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e
        :param context: a single character to score
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class RNNLanguageModel(LanguageModel):
    def __init__(self):
        raise Exception("Implement me")

    def get_next_char_log_probs(self, context):
        raise Exception("Implement me")

    def get_log_prob_sequence(self, next_chars, context):
        raise Exception("Implement me")


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev texts as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNLanguageModel instance trained on the given data
    """
    raise Exception("Implement me")