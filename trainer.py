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
		concept_classifier = self.build_concept_classifier(c[1:])
		result = concept_classifier.predict(_text)
		concept_class = c[1].replace("train/train_", "")

		self.concept_corpus = PlaintextCorpusReader(BASE_PATH, fileids)
		train_concepts = list((' '.join(s), fileid.replace("train/train_", "")) for fileid in fileids for s in concept_corpus.sents(fileid))

		self.vectorizer = TfidfVectorizer(input='content', stop_words=None, ngram_range=(1,2))
		concept_vec = vectorizer.fit_transform(train_concepts)
		target = [tc[1] for tc in train_concepts]

		self.classifier = svm.LinearSVC(class_weight='auto')
        classifier.fit(concept_vec, target)

	def classify_text(self, text):
		pass

def testing():
    print("loading the nondisclosure schema...")
    schema = AgreementSchema()
    schema.load_schema("nondisclosure.ini")
    concepts = schema.get_concepts()
    tupled = []
    print(concepts)

    text = "This is an example of one paragraph."
    for c in concepts:
    	print("Check for this concept: " % c[0])
    	concept_trainer = Trainer(c[1:])
    	result = concept.trainer.classify_text(text)
    	print(result)

    print("the end.")