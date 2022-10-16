# models.py

import numpy as np
import collections
import torch
import torch.nn as nn
from torch import optim

#####################
# MODELS FOR PART 1 #
#####################
class ConsonantDataset(torch.utils.data.Dataset):
    def __init__(self,consonant_examples, vowel_examples, indexer):
        self.indexer = indexer
        self.text = []
        self.labels = []
        self.labels_onehot = []
        for c in consonant_examples:
            x = self.form_input(c)
            #print(x.size())
            y = torch.tensor([1,0],dtype=torch.float)
            self.text.append(x)
            self.labels.append(y)
        for v in vowel_examples:
            x = self.form_input(c)
            y = torch.tensor([0,1],dtype=torch.float)
            self.text.append(x)
            self.labels.append(y)
    def __len__(self):
        return len(self.text)

    def __getitem__(self,idx):
        #text = torch.tensor([])
        #label = torch.tensor([])
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
    def __init__(self, dict_size, input_size, hidden_size, dropout, vocab_index,rnn_type='lstm'):
        super(RNNClassifier, self).__init__()
        self.vocab_index = vocab_index
        self.word_embedding = nn.Embedding(dict_size, input_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.layers = 1
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, num_layers=self.layers, dropout=dropout)
        self.init_weight()
        self.linear = nn.Linear(hidden_size, 2)
        self.debug = False
        self.dd = nn.Dropout(p=0.2)
        self.sm = nn.LogSoftmax(dim=1)
        self.relu = nn.ReLU()
    
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
        #print(x.size())
        #print(input)
        probs = self.forward(x)
        #print(output)
        #print(h[:,-1])
        #probs = self.sm(self.linear(self.dd(h[:,-1])))
        prediction = torch.argmax(probs)
        #print(prediction)
        return prediction

    def init_weight(self):
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
        #nn.init.xavier_uniform_(self.rnn.bias_hh_l0)
        #nn.init.xavier_uniform_(self.rnn.bias_ih_l0)

    def forward(self, input):
        #print(input)
        embedded_input = self.word_embedding(input)
        #print(embedded_input)
        #if(self.debug):
        #    print(embedded_input.size())
        # RNN expects a batch
        #embedded_input = embedded_input.unsqueeze(0)
        #print(embedded_input.size())
        # Note: the hidden state and cell state are 1x1xdim tensor: num layers * num directions x batch_size x dimensionality
        # So we need to unsqueeze to add these 1-dims.
        init_state = (torch.zeros(self.layers, embedded_input.size(1), self.hidden_size).requires_grad_().detach(),
                      torch.zeros(self.layers, embedded_input.size(1), self.hidden_size).requires_grad_().detach())
        #init_state = (torch.from_numpy(np.zeros(self.hidden_size)).unsqueeze(0).unsqueeze(1).float(),
        #              torch.from_numpy(np.zeros(self.hidden_size)).unsqueeze(0).unsqueeze(1).float())
        #print(input)
        #print(init_state[0].size())
        #print(init_state[1].size())
        #print(embedded_input.size())
        #print(init_state.size())
        o,(h,c) = self.rnn(embedded_input, init_state)
        #print(o.size())
        #probs = self.linear(self.relu(o))
        #print(h.size())
        #print(h[:,-1].size())
        probs = self.linear(self.relu(h[:,-1]))
        #print(probs)
        return probs
        # Note: hidden_state is a 1x1xdim tensor: num layers * num directions x batch_size x dimensionality
        #ret = self.classifier(output[:, -1, :])
        #print(ret)
        #return ret


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
    rnn_module = RNNClassifier(dict_size=27, input_size=2, hidden_size=50, dropout=0.0,vocab_index=vocab_index)
    initial_learning_rate = 0.0001
    optimizer = optim.SGD(rnn_module.parameters(), lr = initial_learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 10,factor=0.5)
    loss_func = nn.NLLLoss()
    num_epochs = 100
    train_dataset = ConsonantDataset(train_cons_exs,train_vowel_exs,vocab_index)
    test_dataset = ConsonantDataset(dev_cons_exs,dev_vowel_exs,vocab_index)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=True)
    for epoch in range(num_epochs):
        train_loss = 0.0
        test_loss = 0.0
        rnn_module.train()
        rnn_module.debug = True
        for x,y in train_dataloader:
            #if(rnn_module.debug):
            #print(x.size())
            #print(y.size())
            probs = rnn_module(x)
            #print(h)
            #print(y)
            loss_prob = rnn_module.sm(probs).squeeze(0)
            print(loss_prob)
            print(y)
            loss = loss_func(loss_prob, y.long().squeeze(0))
            rnn_module.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss+= loss.item()*x.size(0)
        rnn_module.eval()
        """for x,y in test_dataloader:
            probs = rnn_module(x)
            loss = loss_func(probs,y)
            test_loss+=loss.item()*x.size(0)
        #print(test_loss/len(test_dataloader.dataset))
        """
        scheduler.step(train_loss/len(train_dataloader.dataset))
        curr_lr = optimizer.param_groups[0]['lr']
        
        #Validation Loss: {test_loss/len(test_dataloader.dataset)}\t \
        print(f'Epoch {epoch}\t \
            Training Loss: {train_loss/len(train_dataloader.dataset)}\t \
            LR:{curr_lr}')
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
