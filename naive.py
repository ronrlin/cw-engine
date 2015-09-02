#!/usr/bin/python
import os
from sklearn import svm
from nltk.corpus.reader.plaintext import CategorizedPlaintextCorpusReader
from nltk.corpus.reader.plaintext import PlaintextCorpusReader

from structure import AgreementSchema
from structure import load_training_data

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
		provisions_all = load_training_data().items()

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
		#stops = set(stopwords.words("english"))
		#words = [words for words in catcorpus.words()]
		#filtered_words = [word for word in word_list if word not in stops]
		# a good place to curate the words_ds which is like the reference db
		self.words_ds = nltk.FreqDist(words.lower().rstrip('*_-. ') for words in self.training_corpus.words())

	def fit2(self):
		import random
		import re
		provisions = load_training_data().items()
		provision_names = [pv[0] for pv in provisions]
		provision_paths = [pv[1] for pv in provisions]

		#provision_names= ['train_waiver', 'confidential_information']
		#provision_paths= ['train/train_waiver', 'train/train_confidential_information']

		self.training_corpus = PlaintextCorpusReader(".", provision_paths)
		texts = [(list(sents), re.sub("train/train_", "", fileid)) for fileid in self.training_corpus.fileids() for sents in self.training_corpus.sents(fileid)]
		# another idea is to make texts by paragraphs, so train on the individual paragraphs, not whole doc
		train_set = [(self.get_features(_doc), _class) for (_doc,_class) in texts]		
		self.classifier = nltk.NaiveBayesClassifier.train(train_set)		


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
			val = self.classifier.classify(self.get_features(d))
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
	aligner.fit2()
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


def main():
	from structure import AgreementSchema
	schema = AgreementSchema()
	schema.load_schema("nondisclosure.ini")

	from naive import NaiveAlignment
	aligner = NaiveAlignment(schema=schema)
	aligner.fit2()
	aligner.show_info()
	#filename = "nda-0000-0014.txt"
	#print("obtain a corpus...")
	#from classifier import build_corpus
	#corpus = build_corpus()
	
	provisions = load_training_data().items()
	provision_names = [pv[0] for pv in provisions]
	provision_paths = [pv[1] for pv in provisions]

	#doc = corpus.raw(filename)
	paras = ['Confidential Information includes, Confidential Information but is not limited to, the following, whether now in existence or hereafter created']
	aligned = aligner.align(paras) # aligned_provisions is a list of tuples
	print(aligned)

	training_corpus = PlaintextCorpusReader(".", provision_paths)

	texts = [(list(sents), re.sub("train/train_", "", fileid)) for fileid in training_corpus.fileids() for sents in training_corpus.sents(fileid)]
	print(texts)
	# another idea is to make texts by paragraphs, so train on the individual paragraphs, not whole doc
	#random.shuffle(texts)
	#train_set = [(self.get_features(_doc), _class) for (_doc,_class) in texts]		
	#self.classifier = nltk.NaiveBayesClassifier.train(train_set)	