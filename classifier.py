#!/usr/bin/python
import os
from sklearn import svm
from nltk.corpus.reader.plaintext import CategorizedPlaintextCorpusReader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

BASE_PATH = "./"
CORPUS_PATH = os.path.join(BASE_PATH, "data/")

class AgreementVectorClassifier(object):
	""" """
	def __init__(self, vectorizer, corpus):
		""" """
		self.corpus = corpus
		self.vectorizer = vectorizer
		self.classifier = None

	def fit(self):
		""" Fit the classifier """
		textcomp = []
		for thisfile in self.corpus.fileids():
			text = self.corpus.raw(thisfile)
			textcomp.append([text])

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
		self.words_ds = nltk.FreqDist(words.lower().rstrip('*_-. ') for words in self.corpus.words())

	def fit(self):
		""" """
		import random
		texts = [(list(self.corpus.words(fileid)), category) for category in self.corpus.categories() for fileid in self.corpus.fileids(category)]
		random.shuffle(texts)
		train_set = [(self.get_features(_doc), _class) for (_doc,_class) in texts]		
		self.classifier = nltk.NaiveBayesClassifier.train(train_set)

	def get_features(self, doc):
		""" 
		Returns the features for this document.
		:param doc: list of strings 
		"""
		doc_words = set(doc)
		word_features = list(self.words_ds)[:3000]
		features = {}
		for word in word_features:
			features['contains(%s)' % word] = (word in doc_words)
		return features

	def classify_data(self, data):
		""" 
		:param data: list of strings
		"""
		return self.classifier.classify(self.get_features(data))

	def show_info(self):
		self.classifier.show_most_informative_features(n=50)

def build_corpus(binary=False, binary_param=None): 
	""" 
	Obtain a corpus, often to be used for training.

	:param binary: boolean so that the corpus uses binary classifications.
	:param binary_param: string which specifies which category to have, category2 is 'other'.

	Returns a corpus, which is categorized using information from the WiserDatabase.
	"""
	import csv
	import os
	from helper import WiserDatabase
	wd = WiserDatabase()
	fileids = list()
	cats = list()
	corpus_details = {}

	if (binary) and (binary_param):
		binary_search = [binary_param]
	else:
		binary_search = ['nondisclosure']	

	for category in binary_search:
		results = wd.fetch_by_category(category)
		for r in results:
			if r is not None:
				fileids.append(r['filename'])
				cats.append(r['category'])
				corpus_details[r['filename']] = r['category']

	categories = wd.get_category_names()
	categories.remove(binary_search[0])
	for category in categories:
		results = wd.fetch_by_category(category)
		for r in results:
			if r is not None:
				fileids.append(r['filename'])
				cats.append("other")
				corpus_details[r['filename']] = r['category']

	print("Loading the training corpus.")
	newvals = {}
	binvals = {}
	i = 0
	for f in fileids:
		newvals[f] = [corpus_details[f]]
		binvals[f] = [cats[i]]
		i = i + 1

	if (binary):
		train_corpus = CategorizedPlaintextCorpusReader(CORPUS_PATH, fileids, cat_map=binvals)
	else:
		train_corpus = CategorizedPlaintextCorpusReader(CORPUS_PATH, fileids, cat_map=newvals)

	return train_corpus

def get_agreement_classifier_v1(train_corpus):
	""" """
	count_vectorizer = CountVectorizer(input='content', stop_words=None, ngram_range=(1,2))
	countClass = AgreementVectorClassifier(count_vectorizer, train_corpus)
	countClass.fit()
	return countClass

def get_agreement_classifier_v2(train_corpus):
	""" """
	tfidf_vectorizer = TfidfVectorizer(input='content', stop_words=None, ngram_range=(1,2))	
	tfidfClass = AgreementVectorClassifier(tfidf_vectorizer, train_corpus)
	tfidfClass.fit()
	return tfidfClass

def get_agreement_classifier_v3(train_corpus):
	""" """
	naiveClass = AgreementNaiveBayesClassifier(train_corpus)
	naiveClass.fit()
	return naiveClass

def testing():
	print("Loading the datastores...")
	from helper import WiserDatabase
	datastore = WiserDatabase()
	print("Build a corpus of documents...")
	corpus = build_corpus(binary=False, binary_param='nondisclosure')
	print("Loading the agreement classifier...")
	classifier = get_agreement_classifier_v3(corpus)
	print("Application ready to load.")
	#agreement_type = classifier.classify_file("/home/obironkenobi/Projects/ContractWiser/small-data/nda-0000-0025.txt")
	doc = corpus.words("nda-0000-0025.txt")
	agreement_type = classifier.classify_data(doc)
	print("The agreement is a %s agreement" % agreement_type)
	#classifier.show_info()
	doc = corpus.words("nda-0000-0001.txt")
	agreement_type = classifier.classify_data(doc)
	print("The agreement is a %s agreement" % agreement_type)

	doc = corpus.words("b9debb6714e6c54e0e3ac431af2215c24784d195418988be65568fd92dfb3fd7")
	agreement_type = classifier.classify_data(doc)
	print("The agreement is a %s agreement" % agreement_type)

def main():
	pass

"""
"""
if __name__ == "__main__":
    pass