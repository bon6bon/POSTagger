from sklearn import metrics
from sklearn.metrics import accuracy_score
from nltk.corpus import conll2000
from sklearn import preprocessing
from classifier.classifier import Classifier
from model.model import Model, Reader
import os


class POSTagger():

    def __init__(self):
        self.model = Model()
        self.model.model_load()
        self.r = Reader()
        self.r.read_corpus()
        self.tagger = Classifier(self.r.train_sents, self.model)

    def evaluate(self, featureset):
        """
        Evaluate the accuracy of the classifer based POS tagger
        featureset: [[features extracted for a word, tag in a gold standard]]
        stdout: accuracy_score
        """
        #sequence, tag = featureset
        gs, labels = [], []
        for s, t in featureset:
            gs.append(t)
            label = self.tagger.choose_tag(s)
            labels.append(label)
            print (t, label)

        assert(len(gs) == len(labels))
        self.write_to_file(labels)
        words =  self.tagger.test(self.r.test_sents, word=True)
        print (accuracy_score(gs, labels))

    def write_to_file(self, labels):
        with open('labels.txt', 'w') as file_handler:
            for label in labels:
                file_handler.write("{}\n".format(label))


if __name__ == '__main__':

    pos_tagger = POSTagger()
    classifier_name = os.path.join("classifier", "classifier_trained")
    vectorizer_name = os.path.join("classifier", "vectorizer_trained")
    if not (os.path.exists(classifier_name) or os.path.exists(vectorizer_name)):
        pos_tagger.tagger.train(pos_tagger.r.train_sents, pos_tagger.tagger.classifier_builder)
        pos_tagger.tagger.persist(classifier_name, vectorizer_name)
    else:
        pos_tagger.tagger.load(classifier_name, vectorizer_name)
    
    featureset = pos_tagger.tagger.test(pos_tagger.r.test_sents)
    #print (featureset)
    pos_tagger.evaluate(featureset)
