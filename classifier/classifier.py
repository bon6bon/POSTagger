import nltk
from nltk.chunk import ChunkParserI, tree2conlltags, conlltags2tree
from gensim.models import Word2Vec
import os
import numpy as np 
import re
from sklearn.externals import joblib
from nltk.classify.naivebayes import NaiveBayesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer

class Classifier(ChunkParserI):

    def __init__(self, train_sents, model, **kwargs):
        """
            self.model: a model to be used for Semantic Space
            self.classifier_builder: a classifier type to train the custom classifier
            self.dictVectorizer: vectoizer for categorical data
        """
        self.model = model
        self.classifier_builder = LogisticRegression() #RandomForestClassifier() or NaiveBayesClassifier()
        self.dictVectorizer = DictVectorizer(sparse=False)

    def feature_detector(self, tokens, index, history):
        """
           tokens: [list of words in a sentence]
           index: current position of a word pointer in a sentence 
           history: [list of POS tages assigned for prevword and prevprevword]

           Return:
           features: [diimension-sized list of doubles representing a word in a semantic space] 
           features_to_vect: {dict of categorical features assigned to a word}
        """
        word = tokens[index]
        if index == 0:
            prevword = prevprevword = None
            prevtag = prevprevtag = None
        elif index == 1:
            prevword = tokens[index-1].lower()
            prevprevword = None
            prevtag = history[index-1]
            prevprevtag = None
        else:
            prevword = tokens[index-1].lower()
            prevprevword = tokens[index-2].lower()
            prevtag = history[index-1]
            prevprevtag = history[index-2]
        if index == len(tokens)-1:
            nextword = None
            nextnextword = None
        elif index == len(tokens)-2:
            nextword =  tokens[index+1].lower()
            nextnextword = None
        else:
            nextword =  tokens[index+1].lower() 
            nextnextword =  tokens[index+2].lower() 

        if re.match('[0-9]+(\.[0-9]*)?|[0-9]*\.[0-9]+$', word):
            shape = 'number'
        elif re.match('\W+$', word):
            shape = 'punct'
        elif re.match('[A-Z][a-z]+$', word):
            shape = 'upcase'
        elif re.match('[a-z]+$', word):
            shape = 'downcase'
        elif re.match('\w+$', word):
            shape = 'mixedcase'
        else:
            shape = 'other'

        features = np.hstack( (self.model.get_vector(prevprevword), self.model.get_vector(prevword), self.model.get_vector(word), self.model.get_vector(nextword), self.model.get_vector(nextnextword)) )

        features_to_vect = {
            'prevtag': prevtag,
            'prevprevtag': prevprevtag,
            #'suffix3': word.lower()[-3:],
            #'suffix2': word.lower()[-2:],
            #'suffix1': word.lower()[-1:],
            'shape': shape,
            }
        return [features, features_to_vect]


    def _corpus_construct(self, tagged_corpus, phase, verbose=True):

        """
        tagged_corpus: [array of sentences with [(word, POS)]]
        phase: string (training / testing)
        return: [[feature vectors in doubles for each word], [{dict of categorical features}], [labels]]
        """
        classifier_corpus = []
        if verbose:
            print ('Constructing ' + phase + ' corpus for classifier.')

        #tag_sents = [tree2conlltags(sent) for sent in tagged_corpus] #needed for CONLL only
        #train_chunks = [[(w,t) for (w,t,c) in sent] for sent in tag_sents] #needed for CONLL only
        train_chunks = tagged_corpus

        for sentence in train_chunks:
            history = []
            untagged_sentence, tags = zip(*sentence)
            for index in range(len(sentence)):
                featureset, features_to_vect = self.feature_detector(untagged_sentence, index, history)
                classifier_corpus.append( (featureset, features_to_vect, tags[index]) )
                history.append(tags[index])

        if verbose and phase == "training":
            print ('Training classifier (%d instances)' % len(classifier_corpus))
        features = [x[0] for x in classifier_corpus]
        labels = [x[2] for x in classifier_corpus]
        features_to_vect = [x[1] for x in classifier_corpus]
        return [features, features_to_vect, labels]

    def train(self, tagged_corpus, classifier_builder):
        """
        tagged_corpus: [array of sentences with [(word, POS)]]
        classifier_builder: Classifier Object
        self.classifier: fitted Classifier
        """
        features, features_to_vect, labels = self._corpus_construct(tagged_corpus, "training")
        transformed_features = self.dictVectorizer.fit_transform(features_to_vect)
        transformed_features = np.nan_to_num(transformed_features)
        features = np.hstack((features, transformed_features))
        self.classifier = classifier_builder.fit(features, labels)
        self.classifier.fit(features,labels)

    def test(self, tagged_corpus, word = False):
        """
        tagged_corpus: [array of sentences with [(word, POS)]]
        classifier_builder: Classifier Object
        Return: zip(feature vectors, GS labels)
        """
        features, features_to_vect, labels = self._corpus_construct(tagged_corpus, "testing")
        if word:
            return labels
        transformed_features = self.dictVectorizer.transform(features_to_vect)
        transformed_features = np.nan_to_num(transformed_features)
        features = np.atleast_2d(np.hstack((features, transformed_features)))
        return zip(features, labels) 

    def choose_tag(self, featureset):
        """
            Assign a label to one feature vector using a loaded / created classifier with its vectorizer 
            Return: str(label)
        """
        label = self.classifier.predict( np.atleast_2d(featureset) )
        #print (label)
        return label

    def parse(self, tagged_sent):
        """
            for CONLL corpus only
        """
        if not tagged_sent: return None
        chunks = self.tagger.tag(tagged_sent)
        return conlltags2tree([(w,t,c) for ((w,t),c) in chunks])

    def chunk_trees2train_chunks(self, chunk_sents):
        """
            for CONLL corpus only
        """
        tag_sents = [tree2conlltags(sent) for sent in chunk_sents]
        return [[((w,t),c) for (w,t,c) in sent] for sent in tag_sents]

    def chunk_trees2train_word_pos(self, chunk_sents):
        """
            for CONLL corpus only
        """
        tag_sents = [tree2conlltags(sent) for sent in chunk_sents]
        return [[(w,t) for (w,t,c) in sent] for sent in tag_sents]

    def conll_tag_chunks(self, chunk_sents):
        """
            for CONLL corpus only

        Convert each chunked sentence to list of (tag, chunk_tag) tuples,
        so the final result is a list of lists of (tag, chunk_tag) tuples.
        >>> from nltk.tree import Tree
        >>> t = Tree('S', [Tree('NP', [('the', 'DT'), ('book', 'NN')])])
        >>> conll_tag_chunks([t])
        """
        tagged_sents = [tree2conlltags(tree) for tree in chunk_sents]
        return [[(t, c) for (w, t, c) in sent] for sent in tagged_sents]

    def persist(self, classifier_name, vectorizer_name):
        """Save trained classifer and its vectorizer"""
        joblib.dump(self.classifier, classifier_name)
        joblib.dump(self.dictVectorizer, vectorizer_name)

    def load(self, classifier_name, vectorizer_name):
        """Load pre-trained classifier and its vectorizer"""
        self.classifier = joblib.load(classifier_name)
        self.dictVectorizer = joblib.load(vectorizer_name)
    