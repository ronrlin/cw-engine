#!/usr/bin/python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.corpus.reader.plaintext import CategorizedPlaintextCorpusReader
from sklearn import svm

import os
import nltk
from nltk.tokenize import word_tokenize
from structure import AgreementSchema
from structure import load_training_data
from structure import get_provision_name_from_file

BASE_PATH = "./"

class Trainer(object):
	""" """

	def __init__(self, fileids=None):
		print(fileids)
		self.concept_corpus = PlaintextCorpusReader(BASE_PATH, fileids)
		train_concepts = list((' '.join(s), fileid.replace("train/train_", "")) for fileid in fileids for s in self.concept_corpus.sents(fileid))

		self.vectorizer = TfidfVectorizer(input='content', stop_words=None, ngram_range=(1,2))
		training_text = [tc[0] for tc in train_concepts]
		concept_vec = self.vectorizer.fit_transform(training_text)
		target = [tc[1] for tc in train_concepts]
		# TODO: consider stripping out train/train_ from target

		self.classifier = svm.LinearSVC(class_weight='auto')
		self.classifier.fit(concept_vec, target)

	def classify_text(self, text):
		data_vectorized = self.vectorizer.transform(text)
		results = self.classifier.predict(data_vectorized)
		return results[0]

def testing():
    print("loading the nondisclosure schema...")
    schema = AgreementSchema()
    schema.load_schema("nondisclosure.ini")
    concepts = schema.get_concepts()
    tupled = []
    print(concepts)

    text = "This is an example of one paragraph."
    for c in concepts:
    	print("Check for this concept: %s" % c[0])
    	paths = c[1].replace(" ", "")
    	path_list = paths.split(',')

    	concept_trainer = Trainer(path_list)
    	result = concept_trainer.classify_text(text)
    	print(result)

    print("the end.")