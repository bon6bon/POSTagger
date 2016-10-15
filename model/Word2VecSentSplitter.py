"""Word2VecSentSplitter is a class for processing raw HTML text into a list of words for further learning"""

import re
import string
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords


class Word2VecSentSplitter():

    @staticmethod
    def review_to_wordlist( sentence, remove_stopwords = False ):
        """ Convert a document to a list of words.
        Returns a list of words.

        """
        # Replace urls with <URL>
        sentence = re.sub(r'(?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’])', '<URL>', sentence) 
        # Strip punctuation
        sentence = re.sub('[%s]' % re.escape(string.punctuation), '', sentence) 
        # Replace numbers with <NUM>
        sentence = re.sub(r"\d+-?(\w+)?", '<NUM>', sentence)
        # Remove HTML
        sentence_text = BeautifulSoup(sentence).get_text()
        # Remove non-letters
        sentence_text = re.sub("[^a-zA-Z]"," ", sentence_text)
        # Convert words to lower case and split them
        words = sentence_text.lower().split()
        # Remove stop words (false by default), optionally
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            words = [w for w in words if not w in stops]
        # Return a list of words
        return(words)

    @staticmethod
    def review_to_sentences( review, tokenizer, remove_stopwords = False ):
        # Split a review into parsed sentences. 
        # Return a list of sentences, where each sentence is a list of words
        # 1. Use the NLTK tokenizer to split the paragraph into sentences
        raw_sentences = tokenizer.tokenize(review.decode('utf8').strip())
        # 2. Go through each sentence
        sentences = []
        for raw_sentence in raw_sentences:
            if len(raw_sentence) > 0:
                # Otherwise, call review_to_wordlist to get a list of words
                sentences.append( Word2VecSentSplitter.review_to_wordlist( raw_sentence, remove_stopwords ))
        # Return the list of sentences (each sentence is a list of words) -> this returns a list of lists
        return sentences

    @staticmethod
    def review_to_sentences_conll( words, tokenizer, remove_stopwords = False ):
        sentence = ''
        for w in words:
            sentence += w + " " 
        # Replace urls with <URL>
        sentence = re.sub(r'(?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’])', '<URL>', sentence) 
        # Strip punctuation
        sentence = re.sub('[%s]' % re.escape(string.punctuation), '', sentence) 
        # Replace numbers with <NUM>
        sentence = re.sub(r"\d+-?(\w+)?", '<NUM>', sentence)
        # Remove HTML
        sentence_text = BeautifulSoup(sentence, 'html.parser').get_text()
        # Remove non-letters
        sentence_text = re.sub("[^a-zA-Z]"," ", sentence_text)
        # Convert words to lower case and split them
        words = sentence_text.lower().split()
        sentences = []
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            words = [w for w in words if not w in stops]
        #print (words)
        sentences.append(words)
        return sentences

