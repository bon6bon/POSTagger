import nltk
import logging
from gensim.models import Word2Vec
from nltk.corpus import conll2000, brown
from nltk.corpus import stopwords
import nltk.data
import os
import numpy as np 
from model.Word2VecSentSplitter import *

class Reader():

    def __init__(self):
        """
            training and test corpora
        """
        self.conll_train = []
        self.conll_test = []
        self.train_sents = []
        self.test_sents = []

    def read_corpus(self):
        """
            assign training and test corpora
        """
        length = len(brown.tagged_sents(tagset='universal'))
        self.train_sents = brown.tagged_sents(tagset='universal')[0:int(length*0.9)]
        self.test_sents = brown.tagged_sents(tagset='universal')[int(length*0.9):]
        # Read data from files
        self.conll_train = conll2000.sents('train.txt') #only when building semantic space
        self.conll_test = conll2000.sents('test.txt') #only when building semantic space

        #unlabeled_text = conll2000.sents()
        #print ("Read %d labeled train reviews, %d labeled test reviews, " "%d unlabeled reviews\n" % ( len(self.conll_train), len(self.conll_test), len(unlabeled_text) ) )
        #self.train_sents = conll2000.chunked_sents('train.txt')
        #self.test_sents = conll2000.chunked_sents('test.txt')

class Model():
    
    def __init__(self):
        self.model_name = os.path.join("space", "GoogleNews-vectors-negative300.bin.gz")
        #self.model_name = os.path.join("space", "300features_5minwords_5context") #small test sematic space
        self.num_features = 300 
        #parameters to create a semantic space
        self.params = {'size': self.num_features, 'min_count': 5, 'workers': 4, 'window': 5, 'sample': 1e-5, 'iter': 10, 'alpha': 0.025, 'negative': 10, 'min_alpha': 1e-2, 'seed': 1}

    def model_load(self):
        """
        Load a semantic space
        """
        #if not (os.path.exists(self.model_name)):
        #    self.model_save()
        print ("Start GoogleNews-vectors-negative300 loading")
        self.model = Word2Vec.load_word2vec_format(self.model_name, binary=True)
        #self.model = Word2Vec.load(self.model_name)
        print ("GoogleNews-vectors-negative300 loaded")
        # Index2word is a list that contains the names of the words in 
        # the model's vocabulary. Convert it to a set, for speed 
        self.index2word_set = set(self.model.index2word)

    def get_vector(self, word):
        """
        word: string 
        Return 
            vecFeature: [lsit of doubles represeting a word in a semantic space] 
        """
        if word in self.index2word_set:
            vecFeature = self.model[word]
        else:
            vecFeature = np.zeros((self.num_features,), dtype="float32")
        return vecFeature

    def model_save(self):
        """
            Save semantic space to self.model_name file
        """
        # Load the punkt tokenizer
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        r = Reader()
        r.read_corpus()

        # ****** Split the labeled and unlabeled training sets into clean sentences
        sentences = []  # Initialize an empty list of sentences
        for sent in r.conll_train:
            sentences += Word2VecSentSplitter.review_to_sentences_conll(sent, tokenizer, remove_stopwords=True)
        # ****** Set parameters and train the word2vec model
        # Import the built-in logging module and configure it so that Word2Vec creates nice output messages
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        
        # Initialize and train the model
        print ("Training Word2Vec model...")
        #print (self.params)
        model = Word2Vec(sentences, **self.params)

        # Calling init_sims makes the model more memory-efficient.
        model.init_sims(replace=True)

        # Save the model for later use. It can be loaded again with Word2Vec.load()
        model.save(self.model_name)