#!/usr/bin/python
import os
from sklearn import svm
from nltk.corpus.reader.plaintext import CategorizedPlaintextCorpusReader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

BASE_PATH = "/home/obironkenobi/Projects/ContractWiser/"
CORPUS_PATH = os.path.join(BASE_PATH, "small-data/")

class AgreementVectorClassifier(object):
	""" """
	def __init__(self, vectorizer, corpus):
		self.corpus = corpus
		self.vectorizer = vectorizer
		self.classifier = None

	def fit(self):
		""" Fit the classifier """
		textcomp = []
		for thisfile in self.corpus.fileids():
			text = self.corpus.raw(thisfile)
			text = ' '.join([text])
			textcomp.append(text)

		train_vector = self.vectorizer.fit_transform(textcomp)
		self.classifier = svm.LinearSVC(class_weight='auto')
		self.classifier.fit(train_vector, list(self.corpus._map.values()))

	def classify_file(self, filename):
		fh = open(filename, 'r')
		x = fh.read()
		fh.close()
		dtm_test = self.vectorizer.transform([x])
		results = self.classifier.predict(dtm_test)
		return results[0]

	def classify_data(self, data):
		data_vectorized = self.vectorizer.transform([data])
		results = self.classifier.predict(data_vectorized)
		return results[0]		

class AgreementNaiveBayesClassifier(object):
	""" """
	def __init__(self, corpus):
		self.corpus = corpus
		self.classifier = None
		self.words_ds = nltk.FreqDist(words.lower() for words in self.corpus.words())

	def fit(self):
		""" """
		import random
		texts = [(list(self.corpus.words(fileid)), category) for category in self.corpus.categories() for fileid in self.corpus.fileids(category)]
		random.shuffle(texts)
		train_set = [(self.get_features(_doc), _class) for (_doc,_class) in texts]		
		self.classifier = nltk.NaiveBayesClassifier.train(train_set)

	def get_features(self, doc):
		""" """
		doc_words = set(doc)
		word_features = list(self.words_ds)[:3000]
		features = {}
		for word in word_features:
			features['contains(%s)' % word] = (word in doc_words)
		return features

	def classify_file(self):
		""" """
		pass

def get_agreement_corpus(): 
	import csv
	import os
	from helper import WiserDatabase

	wd = WiserDatabase()
	category_names = wd.get_category_names()

	fileids = list()
	cats = list()
	corpus_details = {}

	categories = ['convertible']
	for category in categories:
		results = wd.fetch_by_category(category)
		for r in results:
			if r is not None:
				fileids.append(r['filename'])
				cats.append(r['category'])
				corpus_details[r['filename']] = r['category']

	categories = wd.get_category_names()
	categories.remove('convertible')
	for category in categories:
		results = wd.fetch_by_category(category)
		for r in results:
			if r is not None:
				fileids.append(r['filename'])
				cats.append("other")
				corpus_details[r['filename']] = r['category']

	print("Loading the training corpus.")
	newvals = {}
	for f in fileids:
		newvals[f] = corpus_details[f]

	train_corpus = CategorizedPlaintextCorpusReader(CORPUS_PATH, fileids, cat_map=newvals)
	return train_corpus

def get_agreement_classifier_v1(train_corpus):
	count_vectorizer = CountVectorizer(input='content', stop_words=None, ngram_range=(1,2))
	countClass = AgreementVectorClassifier(count_vectorizer, train_corpus)
	countClass.fit()
	return countClass

def get_agreement_classifier_v2(train_corpus):
	tfidf_vectorizer = TfidfVectorizer(input='content', stop_words=None, ngram_range=(1,2))	
	tfidfClass = AgreementVectorClassifier(tfidf_vectorizer, train_corpus)
	tfidfClass.fit()
	return tfidfClass

def get_agreement_classifier_v3(train_corpus):
	naiveClass = AgreementNaiveBayesClassifier(train_corpus)
	naiveClass.fit()
	return naiveClass

def main():
	pass

"""
"""
if __name__ == "__main__":
    pass