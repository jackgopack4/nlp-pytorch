# models.py

from xmlrpc.client import Boolean
from sentiment_data import *
from utils import *
import string
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.util import bigrams
import random

from collections import Counter

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def __init__(self,indexer: Indexer):
        nltk.download('stopwords')
        # stopwords downloaded from nltk corpus site
        self.stopwords = set(stopwords.words('english'))
        self.indexer = indexer
        self.vocab = Counter()
        self.invalidcharacters = set(string.punctuation)
        self.invalidcharacters.update(set(string.digits))
        self.invalidcharacters.add('\'')
        self.invalidcharacters.add('\'')
        self.invalidcharacters.add('\'')
        self.invalidcharacters.add('-')
        self.invalidcharacters.add(',')
        random.seed(69,420)
        # want to have featurizer as Beam?
    
    def get_indexer(self):
        return self.indexer
        #raise Exception("Don't call me, call my subclasses")
    
    def get_vocab(self) -> Counter:
        return self.vocab

    def is_unigram(self) -> Boolean:
        raise Exception("Don't call me, call my subclasses")

    def is_bigram(self) -> Boolean:
        raise Exception("Don't call me, call my subclasses")

    def is_bettergram(self) -> Boolean:
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")
    
    def build_vocab(self, examples: List[SentimentExample]):
        raise Exception("Don't call me, call my subclasses")

class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """"""
    def __init__(self, indexer: Indexer):
        nltk.download('stopwords')
        # stopwords downloaded from nltk corpus site
        self.stopwords = set(stopwords.words('english'))
        self.indexer = indexer
        self.vocab = Beam(1000)
        #raise Exception("Must be implemented")
    """
    def is_unigram(self) -> Boolean:
        return True

    def is_bigram(self) -> Boolean:
        return False

    def is_bettergram(self) -> Boolean:
        return False
    
    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        sentencelower = [x.lower() for x in sentence]
        feats = Counter()
        for w in sentencelower:
            if w in self.vocab.keys():
                feats.update([w])
            else:
                if add_to_indexer:
                    # check if it has invalid characters. if not, add to vocab
                    if not (any(char in self.invalidcharacters for char in w)):
                        feats.update([w])
                        self.vocab.update([w])

        return feats

    def build_vocab(self, examples: List[SentimentExample],size:int):
        # reference: https://www.codespeedy.com/detect-if-a-string-contains-special-characters-or-not-in-python/#:~:text=%20string.punctuation%20to%20detect%20if%20a%20string%20contains,in%20x%29%3A%20print%20%28%22invalid%22%29%20else%3A%20print%28%22valid%22%29%20Output%3A%20valid
        vocabCounter = Counter()
        i:int = 0
        for e in examples:
            e.words = [x.lower() for x in e.words]
            e.words = [word for word in e.words if word not in self.stopwords]
            #print(e.words)
            for w in e.words:
                #worig = w
                #w = w.lower()
                for char in self.invalidcharacters:
                    if char in w:
                        e.words.remove(w)
                        break
                #if any(char in self.invalidcharacters for char in w):
                #    e.words.remove(w)
            vocabCounter.update(e.words)
        #vocabCounter.update(examples[i].words)
        self.vocab = vocabCounter
        #for key, value in vocabCounter.items():
        #    self.vocab.add(key,value)
        #print(self.vocab)
        #print("vocab size: %d"%len(self.vocab.items()))
        for f in self.vocab.most_common(size):
            self.indexer.add_and_get_index(f[0])
        #print(self.indexer)




class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def is_unigram(self) -> Boolean:
        return False

    def is_bigram(self) -> Boolean:
        return True

    def is_bettergram(self) -> Boolean:
        return False

    def build_vocab(self, examples: List[SentimentExample],size:int):
        vocabCounter = Counter()
        i:int = 0
        for e in examples:
            for w in e.words:
                if any(char in self.invalidcharacters for char in w):
                    e.words.remove(w)
            e.words = [x.lower() for x in e.words]
            #e.words = [word for word in e.words if word not in self.stopwords]
        for e in examples:
            bgrm = list(bigrams(e.words))
            bgrm = [bg for bg in bgrm if not (bg[0] in self.stopwords and bg[1] in self.stopwords)]
            vocabCounter.update(bgrm)
        self.vocab = vocabCounter
        #for key, value in vocabCounter.items():
        #    self.vocab.add(key,value)
        #print(self.vocab)
        #print("vocab size: %d"%len(self.vocab.items()))
        for f in self.vocab.most_common(size):
            self.indexer.add_and_get_index(f[0])
        #print(self.indexer)
    #def __init__(self, indexer: Indexer):
    #    raise Exception("Must be implemented")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        sentencelower = [x.lower() for x in sentence]
        feats = Counter()
        for b in bigrams(sentencelower):
            if b in self.vocab.keys():
                feats.update([b])
            else:
                if add_to_indexer:
                    # check if it has invalid characters. if not, add to vocab
                    if (not any(char in self.invalidcharacters for char in b[0]) and
                        not any(char in self.invalidcharacters for char in b[1])):
                        feats.update([b])
                        self.vocab.update([b])
        return feats

class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def is_unigram(self) -> Boolean:
        return False

    def is_bigram(self) -> Boolean:
        return False

    def is_bettergram(self) -> Boolean:
        return True

    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self,featureExtractor:FeatureExtractor,weights,train_exs:List[SentimentExample],size:int):
        #raise Exception("Must be implemented")
        self.a:int = 1
        self.size=size
        weights = np.zeros(self.size)
        self.weights = weights
        self.featureExtractor = featureExtractor

        self.featureExtractor.build_vocab(train_exs,self.size)

    def predict(self, sentence: List[str]) -> int:
        # extract features
        feats:Counter = self.featureExtractor.extract_features(sentence)
        # get indices of those features
        index:Indexer = self.featureExtractor.get_indexer()
        weightsum:float = 0.0
        # add up weights from weight vector
        for f in feats.items():
            if index.contains(f[0]):
                i = index.add_and_get_index(f[0],False)
                weightsum += self.weights[i]*f[1]
        # return prediction (if >0, predict positive, else negative)
        if weightsum>0:
            return 1
        else:
            return 0

class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def predict(self, sentence: List[str]) -> int:
        feats:Counter = self.featureExtractor.extract_features(sentence)
        index:Indexer = self.featureExtractor.get_indexer()
        weightsum:float = 0.0
        for f in feats.items():
            if index.contains(f[0]):
                i = index.index_of(f[0])
                weightsum += self.weights[i]*f[1]
        sig = self.sigmoid(weightsum)
        if sig > 0.5:
            return 1
        else:
            return 0

    # adapted from https://medium.com/analytics-vidhya/logistic-regression-algorithm-from-scratch-with-python-447b17c7502
    def sigmoid(self,z):
        h = 1 / (1 + np.exp(-z))
        return h
    
    def loss(self,sentence: List[str]) -> float:
        feats:Counter = self.featureExtractor.extract_features(sentence)
        index:Indexer = self.featureExtractor.get_indexer()
        weightsum:float = 0.0
        for f in feats.items():
            if index.contains(f[0]):
                i = index.index_of(f[0])
                weightsum += self.weights[i]*f[1]
        sig = self.sigmoid(weightsum)
        return sig
    def gradient(self,theta,X,y):
        m = X.shape[0]
        temp = self.sigmoid(np.dot(X,theta)) - y
        grad = np.dot(temp.T,X).T / m
        return grad
    
    def __init__(self,featureExtractor:FeatureExtractor,weights,train_exs:List[SentimentExample],size:int):
        self.size = size
        #weights = np.zeros(size)
        self.weights = weights
        self.featureExtractor = featureExtractor
        self.featureExtractor.build_vocab(train_exs,size)
        m = len(train_exs)
        


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    size:int
    if(feat_extractor.is_unigram()):
        size = 15000
    elif(feat_extractor.is_bigram()):
        size = 52000
    else:
        size = 5000
    weights = np.zeros(size)
    classifier = PerceptronClassifier(feat_extractor,weights,train_exs,size)
    trainset_length = len(train_exs)
    amount_incorrect = trainset_length
    index = feat_extractor.get_indexer()
    if(feat_extractor.is_unigram()):
        lr:float = .25
        target = 4
    elif(feat_extractor.is_bigram()):
        lr:float = 0.75
        target = 38
    else:
        lr:float = 1
        target = 0

    epochs:int = 30
    counter:int = 0
    while(amount_incorrect>target):
        if(50==counter):
            lr /= 2
        if 100==counter:
            lr /= 2
        if 200==counter:
            lr /= 2
        amount_incorrect = trainset_length
        random.shuffle(train_exs)
        for ex in train_exs:
            label = ex.label
            if label != classifier.predict(ex.words):
                feats = feat_extractor.extract_features(ex.words)
                feat_items = list(feats.items())
                feat_keys = list(feats.keys())
                for f in feat_items:
                    if(index.contains(f[0])):
                        j = index.add_and_get_index(f[0],False)
                        #if(f[1]>2):
                        #    print("there are %d occurrences of %s"%(f[1],f[0]))
                        if(ex.label == 1):
                            classifier.weights[j]+=lr*f[1]
                        else:
                            classifier.weights[j]-=lr*f[1]
            else:
                amount_incorrect-=1 
        counter+=1
    print("trainer ran for %d epochs"%counter)
    return classifier
    #raise Exception("Must be implemented")


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    if feat_extractor.is_unigram():
        epochs = 150
        alpha = 100
        size = 13528
        target = 100
    elif feat_extractor.is_bigram():
        epochs = 200
        size = 61983
        alpha:float = 100
        target = 20
    else:
        size = 13000
    weights = np.random.uniform(-0.01,0.01,size) # initialize to random values between -0.01 and 0.01
    print(weights)
    classifier = LogisticRegressionClassifier(feat_extractor,weights,train_exs,size)
    index = feat_extractor.get_indexer()
    m = len(train_exs)
    print("training set length: ",m)
    print("vocab size: ", len(feat_extractor.get_vocab()))
    weightchange = np.zeros(size)
    weightdelta = 0.0
    amount_incorrect = m
    i=0
    while(amount_incorrect>target):
        amount_incorrect=m        
        print("epoch: ",i)
        if feat_extractor.is_unigram():
            if i % 40 == 0 and i != 0:
                alpha /= 1.25
                print("learning rate ",alpha)
        elif feat_extractor.is_bigram():
            if i % 35 == 0 and i != 0:
                alpha /= 1.5
                print("learning rate ", alpha)
                print("weights: ",classifier.weights)
        prevweightchange = weightchange
        prevweightdelta = weightdelta
        weightchange = np.zeros(size)
        #random.shuffle(train_exs)
        for ex in train_exs:
            label = ex.label
            weightsum:float = 0.0
            feats = feat_extractor.extract_features(ex.words)
            feat_items = list(feats.items())
            feat_keys = list(feats.keys())
            feat_values = list(feats.values())
            for f in feat_items:
                if index.contains(f[0]):
                    j = index.index_of(f[0])
                    jweight = classifier.weights[j]
                    feat_frequency = f[1]
                    weightsum += jweight*feat_frequency
            sig = classifier.sigmoid(weightsum)
            if(sig>0.5 and label == 1) or (sig<=0.5 and label == 0):
                amount_incorrect-=1
            #sig = classifier.loss(ex.words)
            temp = sig-label
            feat_len = len(feat_items)
            for f in feat_items:
                if(index.contains(f[0])):
                    j = index.index_of(f[0])
                    tempupdate = temp*f[1]
                    if(j == 61982):
                        print("weight delta before update: ",weightchange[j])
                        print("classifier weight: ",classifier.weights[j])
                        print("amount to add to weightchange: ",-alpha*(1/m)*tempupdate)
                    weightchange[j]-=alpha*(1/m)*tempupdate
        #weightchange = weightchange
        weightdelta = np.linalg.norm(weightchange-prevweightchange)
        print("weightdelta ", weightdelta)
        print("change from last time ",weightdelta-prevweightdelta)
        #print(weightchange)
        classifier.weights+=weightchange
        print("amount incorrect: ",amount_incorrect)
        i+=1
    return classifier

    #raise Exception("Must be implemented")


def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model