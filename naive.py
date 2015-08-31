#!/usr/bin/python
import os
from sklearn import svm
from nltk.corpus.reader.plaintext import CategorizedPlaintextCorpusReader
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from structure import AgreementSchema

import nltk

BASE_PATH = "./"
CORPUS_PATH = os.path.join(BASE_PATH, "train/")

DATA_PATH = "/home/obironkenobi/Projects/ContractWiser/"
AGREEMENT_PATH = os.path.join(DATA_PATH, "small-data/")

class NaiveAlignment(object):

	def __init__(self, schema=None):
		self.schema = schema
		self.classifier = None

		schema = AgreementSchema()
		schema.load_schema("nondisclosure.ini")

		print("Load %s agreement training provisions" % schema.get_agreement_type())
		provisions_in = schema.get_provisions()
		provisions_out = schema.list_provisions()

		# provisions_reqd is a tuple (provision_name, provision_path)
		training_file_names = [p[1] for p in provisions_in]
		provision_names = [p[0] for p in provisions_in]       

		self.training_corpus = PlaintextCorpusReader(BASE_PATH, training_file_names)
		print("Corpus is loading %s files" % str(len(self.training_corpus.fileids())))

		from helper import WiserDatabase
		self.datastore = WiserDatabase()
		records = self.datastore.fetch_by_category(schema.get_agreement_type())
		fileids = [r['filename'] for r in records]
		self.agreement_corpus = PlaintextCorpusReader(AGREEMENT_PATH, fileids)
		print("Agreement Corpus of type %s has been loaded." % schema.get_agreement_type())

		# Based on the schema, you load a corpus
		self.words_ds = nltk.FreqDist(words.lower().rstrip('*_-. ') for words in self.training_corpus.words())

	def fit(self):
		""" """
		import random
		import re
		# might be a good place to remove punctuation and obviously bad words for fitting.
		texts = [(list(self.training_corpus.words(fileid)), re.sub("train/train_", "", fileid)) for fileid in self.training_corpus.fileids()]

		# another idea is to make texts by paragraphs, so train on the individual paragraphs, not whole doc
		random.shuffle(texts)
		train_set = [(self.get_features(_doc), _class) for (_doc,_class) in texts]		
		self.classifier = nltk.NaiveBayesClassifier.train(train_set)		

	def align(self, data):
		"""
		:param data: list of strings, where each string is a paragraph or sentence or block.
		"""
		output = []
		for d in data:
			val = self.classifier.classify(self.get_features(tokenize(d)))
			output.append(val)

		#return output
		return list(zip(data, list(output)))


	def get_features(self, doc=['list', 'of', 'strings']):
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

	def show_info(self, n=50):
		self.classifier.show_most_informative_features(n=n)


def tokenize(content):
    """ 
    :param content: is a string that is to be tokenized
    return list of strings
    """
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    return tokenizer.tokenize(content)

def testing():
	from structure import AgreementSchema
	schema = AgreementSchema()
	schema.load_schema("nondisclosure.ini")

	aligner = NaiveAlignment(schema=schema)
	aligner.fit()
	aligner.show_info()
	#filename = "nda-0000-0014.txt"
	#print("obtain a corpus...")
	#from classifier import build_corpus
	#corpus = build_corpus()

	#doc = corpus.raw(filename)
	paras = ['Confidential Information includes, Confidential Information but is not limited to, the following, whether now in existence or hereafter created']
	aligned = aligner.align(paras) # aligned_provisions is a list of tuples
	print(aligned)
	return(aligner)

"""
Bypass main
"""
if __name__ == "__main__":
    pass