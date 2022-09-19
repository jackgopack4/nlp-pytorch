# models.py

from cmath import log
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self,reviews:List[SentimentExample], embeddings:WordEmbeddings,classes:int):
        # take in reviews as List of SentimentExamples and embeddings as WordEmbeddings type
        self.word_embeddings = embeddings
        self.text = []
        self.labels = []
        self.labels_onehot = []
        for r in reviews:
            x = self.form_input(r.words)
            y = torch.tensor(r.label,dtype=torch.int64)
            self.text.append(x)
            self.labels.append(y)
            y_onehot = torch.zeros(classes,dtype=torch.float)
            y_onehot.scatter_(0, y, 1.0)
            self.labels_onehot.append(y_onehot)
    
    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = torch.tensor([])
        label = torch.tensor([])
        label_onehot = torch.tensor([])
        if torch.is_tensor(idx):
            idx = idx.tolist()
        try:
            object_iterator = iter(idx)
        except TypeError as te:
            #print("in single mode")
            text = self.text[idx]
            label = self.labels[idx]
            label_onehot = self.labels_onehot[idx]
        else:
            #print("in batch mode")
            for i in idx:
                torch.stack([text,self.text[i]])
                torch.stack([label,self.labels[i]],dim=1)
                torch.stack([label_onehot,self.labels_onehot[i]])
        return text,label,label_onehot

    def form_input(self, x) -> torch.Tensor:
        l = len(x)
        embed_length = self.word_embeddings.get_embedding_length()
        temp = [0] * embed_length
        for w in x:
            temp+=self.word_embeddings.get_embedding(w)
        temp = [i/l for i in temp]
        ret = torch.tensor(temp,dtype=torch.float)
        return ret    

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
        self.dd = nn.Dropout(p=0.1)
        self.g = nn.ReLU()
        self.W = nn.Linear(hid, out)
        self.log_softmax = nn.LogSoftmax(dim=0) # if we use CrossEntropy loss don't need, only with NLL
        # Initialize weights according to a formula due to Xavier Glorot.
        nn.init.kaiming_uniform_(self.V.weight) # look up ways to initialize
        nn.init.kaiming_uniform_(self.W.weight)
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
        return self.log_softmax(self.W(self.g(self.dd(self.V(x)))))
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
        #print(ex_words)
        with torch.no_grad():
            x = self.form_input(ex_words)
            log_probs = self(x)
        #print(log_probs)
        prediction = torch.argmax(log_probs)
        #print("prediction: %f"%prediction)
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
    #batch_size = args.batch_size
    batch_size = 16
    num_examples = len(train_exs)
    feat_vec_size = word_embeddings.get_embedding_length()
    #embedding_size = args.hidden_size
    embedding_size = 200
    train_examples = int(np.floor(num_examples*0.90))
    valid_examples = num_examples-train_examples
    num_classes = 2
    random.seed(69)
    random.shuffle(train_exs)
    torch.manual_seed(69)
    train_dataset = SentimentDataset(train_exs[0:train_examples:],word_embeddings,num_classes)
    valid_dataset = SentimentDataset(train_exs[train_examples::],word_embeddings,num_classes)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    # num_epochs on commandline
    num_epochs = 100
    #num_epochs = args.num_epochs
    DAN = NeuralSentimentClassifier(feat_vec_size,embedding_size,num_classes,word_embeddings)
    
    #initial_learning_rate = args.lr 
    initial_learning_rate = 0.0005
    optimizer = optim.Adam(DAN.parameters(), lr = initial_learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 10,factor=0.5)
    loss_func = nn.NLLLoss()

    """
    m = nn.LogSoftmax(dim=1)
    loss = nn.NLLLoss()
    # input is of size N x C = 3 x 5
    input = torch.randn(3, 5, requires_grad=True)
    # each element in target has to have 0 <= value < C
    target = torch.tensor([1, 0, 4])
    print("input: %s"%input)
    print("target: %s"%target)
    output = loss(m(input), target)
    print("output: %s"%output)
    output.backward()
    """

    for epoch in range(0,num_epochs):
        train_loss = 0.0
        valid_loss = 0.0
        DAN.train()
        for x,y,y_onehot in train_dataloader:
            #print("x: %s"%x)
            #print("y: %s"%y)
            #print("y_onehot: %s"%y_onehot)
            log_probs = DAN(x)
            #print(log_probs)
            #print(y_onehot.long())
            loss = loss_func(log_probs,y.long())
            DAN.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*x.size(0)
        DAN.eval()
        for x,y,y_onehot in valid_dataloader:
            log_probs = DAN(x)
            loss = loss_func(log_probs,y.long())
            valid_loss += loss.item()*x.size(0)
        scheduler.step(valid_loss/valid_examples)
        curr_lr = optimizer.param_groups[0]['lr']
        #print(f'Epoch {epoch}\t \
        #    Training Loss: {train_loss/train_examples}\t \
        #    Validation Loss: {valid_loss/valid_examples}\t \
        #    LR:{curr_lr}')
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

