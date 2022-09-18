# models.py

from cmath import log
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class NeuralSentimentClassifier(SentimentClassifier, nn.Module):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)

    Defines the core neural network for doing multiclass classification over a single datapoint at a time. This consists
    of matrix multiplication, tanh nonlinearity, another matrix multiplication, and then
    a log softmax layer to give the ouputs. Log softmax is numerically more stable. If you take a softmax over
    [-100, 100], you will end up with [0, 1], which if you then take the log of (to compute log likelihood) will
    break.

    The forward() function does the important computation. The backward() method is inherited from nn.Module and
    handles backpropagation.
    """
    def __init__(self, inp:int, hid:int, out:int, word_embeddings:WordEmbeddings):
        """
        Constructs the computation graph by instantiating the various layers and initializing weights.

        :param inp: size of input (integer)
        :param hid: size of hidden layer(integer)
        :param out: size of output (integer), which should be the number of classes
        """
        super(NeuralSentimentClassifier, self).__init__()
        self.V = nn.Linear(inp, hid)
        # self.g = nn.Tanh()
        self.g = nn.ReLU()
        self.W = nn.Linear(hid, out)
        self.log_softmax = nn.LogSoftmax(dim=0) # if we use CrossEntropy loss don't need, only with NLL
        # Initialize weights according to a formula due to Xavier Glorot.
        nn.init.xavier_uniform_(self.V.weight) # look up ways to initialize
        nn.init.xavier_uniform_(self.W.weight)
        # Initialize with zeros instead
        # nn.init.zeros_(self.V.weight)
        # nn.init.zeros_(self.W.weight)    
        # self.word_embeddings
        self.word_embeddings = word_embeddings
        # self.embeddings = get_initialized_embedding_layer()
        self.embeddings = self.word_embeddings.get_initialized_embedding_layer()
    
    # def forward() -> from nn.Module
    def forward(self, x):
        """
        Runs the neural network on the given data and returns log probabilities of the various classes.

        :param x: a [inp]-sized tensor of input data
        :return: an [out]-sized tensor of log probabilities. (In general your network can be set up to return either log
        probabilities or a tuple of (loss, log probability) if you want to pass in y to this function as well
        """
        return self.log_softmax(self.W(self.g(self.V(x))))
    # def predict() from SentimentClassifier
        # get sentence embedding for ex_words (1)
        # return torch.argmax of the forward function return
        # self.eval()
    # def predict_all can be used from SentimentClassifier or edit for batching
    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        self.eval()
        x = self.form_input(ex_words)
        log_probs = self.forward(x)
        prediction = torch.argmax(log_probs)
        return prediction

    # consider forming the inputs here, instead of in train function
    # def formInput() returns [1, 300], [1, 50], [100, 300]
        # (1) standard - loop through every word in a sentence, get embedding, and avg. out to get embedding for the sentence
        # use self.word)embeddings.get_embedding(word)
        # (2) batched + efficiency - do this if you pre-process data into indices
        # use self.embeddings([int for int in batch])
        # use torch.mean across correct dimension to get sentence embedding for each example in batch
    # return average embeddings and possible formatted labels as well


    def form_input(self, x) -> torch.Tensor:
        l = len(x)
        embed_length = self.word_embeddings.get_embedding_length()
        temp = [0] * embed_length
        for w in x:
            temp+=self.word_embeddings.get_embedding(w)
        temp = [i/l for i in temp]
        ret = torch.tensor(temp,dtype=torch.float)
        return ret


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    # initialize variables and hyperparameters (input size, batch, hidden, output, lr, epochs)
    # initialize NSC - add inputs to the init() in the NSC - pass everything you think may be necessary, embeddings, sizes
    # initialize loss, optimizer, format data if needed
    # nn.NLLLoss()
    # nn.CrossEntropy() includes softmax, otherwise put LogSoftmax() in forward function
    # formatting data - pre-prepare indices in the word embeddings word_indexer
    feat_vec_size = word_embeddings.get_embedding_length()
    embedding_size = 10
    num_examples = len(train_exs)
    num_classes = 2
    embed_dataset = torch.zeros([num_examples,feat_vec_size],dtype=torch.float)
    #print(embed_dataset)
    embed_labels = torch.zeros([num_examples],dtype=torch.float)
    i=0
    for e in train_exs:
        l = len(e.words)
        temp = [0] * feat_vec_size
        for w in e.words:
            embeddings = word_embeddings.get_embedding(w)
            #print(embeddings)
            #print('embedding length %d'%(len(embeddings)))
            temp += embeddings
        temp = [i/l for i in temp]
        embed_dataset[i] = torch.tensor(temp,dtype=torch.float)
        embed_labels[i] = e.label
        i+=1
    #print(embed_dataset)
    #print(embed_labels)
    num_epochs = 10
    DAN = NeuralSentimentClassifier(feat_vec_size,embedding_size,num_classes,word_embeddings)
    initial_learning_rate = 0.1
    optimizer = optim.Adam(DAN.parameters(), lr = initial_learning_rate)
    for epoch in range(0,num_epochs):
        ex_indices = [i for i in range(0,num_examples)]
        random.shuffle(ex_indices)
        total_loss = 0.0
        for idx in ex_indices:
            x = embed_dataset[idx]
            y = embed_labels[idx].long()
            y_onehot = torch.zeros(num_classes,dtype=torch.long)
            # scatter will write the value of 1 into the position of y_onehot given by y
            y_onehot.scatter_(0, torch.from_numpy(np.asarray(y,dtype=np.int64)), 1)
            #print(x)
            #print(y)
            DAN.zero_grad()
            log_probs = DAN.forward(x).float()
            #print(log_probs)
            loss = nn.NLLLoss()
            output = loss(log_probs,y_onehot)
            total_loss += output
            output.backward()
            optimizer.step()
    return DAN

    # LATER ON/OPTIMIZATION - do any padding or truncation need for my implementation
    # OPTIONAL - look up and utilize:
    # torch.utils.data.TensorDataset(x,y)
    # torch.utils.data.DataLoader(TensorDataset, batch_size, shuffle)

    # double for loop
     # model.train
    # Inside for loop basic layout
    # form inputs if needed (x,y)
    # zero the gradient
    # forward()
    # loss
    # backward()
    # step optimizer

    # return the model
    #raise NotImplementedError

