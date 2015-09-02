#!/usr/bin/python
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus.reader.plaintext import PlaintextCorpusReader

from helper import WiserDatabase
from structure import AgreementSchema
from alignment import Alignment

"""
CorpusStatistics(corpus)
AgreementStatistics(tupleized)
ProvisionStatistics()

The idea is to create instances of the objects above, which write data to the datastore.

function compute(...)
"""
BASE_PATH = "./"

class ProvisionStatistics(object):
	""" """
	def __init__(self):
		# load the train/train_* files into a corpus
		schema = AgreementSchema()
		self.provisions = schema.list_provisions()
		# provisions is a tuple (provision_name, provision_path)
		training_file_names = [p[1] for p in self.provisions]
		self.provision_names = [p[0] for p in self.provisions]
		self.corpus = PlaintextCorpusReader(BASE_PATH, training_file_names)

	def calculate_similarity(self, provisions):
		""" 
		:param provisions: is a list of strings 
		"""
		# vectorize the docs in the corpus
		vect = TfidfVectorizer(min_df=1)
		tfidf = vect.fit_transform(provisions)
		matrix = (tfidf * tfidf.T).A
		similarity_avg = sum(matrix[0]) / len(matrix[0])
		return similarity_avg

	def calculate_complexity(self):
		import numpy as np
		values = []
		for fileid in self.corpus.fileids():
			docsents = self.corpus.sents(fileid)
			text_blocks = [blah for blah in docsents]

			for thistext in text_blocks:
				text = " ".join(thistext)
				character_count = len(text)
				word_count = len(word_tokenize(text))
				sent_count = len(sent_tokenize(text))
				gulpease = 89 - 10 * (character_count/word_count) + 300 * (sent_count/word_count)
				values.append(gulpease)

		return np.mean(values)

class CorpusStatistics(object):
	""" """
	def __init__(self, corpus):
		self.corpus = corpus

	def calculate_similarity(self, category=None):
		""" 
		Calculates a global doc-to-doc similarity average for a corpus of interest.
		Similarity may be computed relative to reference subset of type 'category'.

		:params category: string that represents a category

		Returns the average similarity across documents.
		"""
		# load in the corpus and look the docs
		docs = [self.corpus.raw(fileid) for fileid in self.corpus.fileids(category)]
		# vectorize the docs in the corpus
		vect = TfidfVectorizer(min_df=1)
		tfidf = vect.fit_transform(docs)
		matrix = (tfidf * tfidf.T).A
		similarity_avg = sum(matrix[0]) / len(matrix[0])
		return similarity_avg

	def calculate_complexity(self, category=None):
		"""
		Calculates a global doc-to-doc complexity average for a corpus of interest.
		complexity may be computed relative to reference subset of type 'category'.

		:params category: string that represents a category

		Returns the average complexity across documents(category).
		"""
		import numpy as np
		docs = [self.corpus.raw(fileid) for fileid in self.corpus.fileids(category)]
		values = []
		for text in docs:
			character_count = len(text)
			word_count = len(word_tokenize(text))
			sent_count = len(sent_tokenize(text))
			gulpease = 89 - 10 * (character_count/word_count) + 300 * (sent_count/word_count)
			values.append(gulpease)
		return np.mean(values)

class AgreementStatistics(object):
	""" """
	def __init__(self, tupleized, raw=None):
		""" """
		self.tupleized = tupleized
		self.raw = raw
		self.output = None

	def get_output(self):
		""" 
		This function builds the document that will be kept in the proper datastore.
		Returns a dict that represents statistics about an agreement.
		"""
		return self.output

	def calculate_stats(self):
		""" """
		word_count = 0
		parameters = {}

		doc = [e[0] for e in self.tupleized]
		doc = " ".join(doc)
		parameters['doc_gulpease'] = self.calculate_complexity(doc)
		parameters["word_count"] = len(word_tokenize(doc))
		parameters["para_count"] = len(self.tupleized)

		for (_block, _type) in self.tupleized:
			words = word_tokenize(_block)
			word_count = word_count + len(words)
			parameters["has_" + _type] = True
			parameters[_type + "_gulpease"] = self.calculate_complexity(_block)

		self.output = parameters
		return parameters

	def calculate_complexity(self, text):
		""" 
		Function that computes complexity.
		:param text: string 
		Returns a score from 0 to 100
		"""
		character_count = len(text)
		word_count = len(word_tokenize(text))
		sent_count = len(sent_tokenize(text))
		gulpease = 89 - 10 * (character_count/word_count) + 300 * (sent_count/word_count)
		return gulpease

	def calculate_similarity(self, text, corpus):
		""" 
		This SHOULD compare a given text to a reference corpus.
		:param text: string of text
		:param b: string of text
		"""
		docs = [corpus.raw(fileid) for fileid in corpus.fileids()]
		docs.append(text)
		# vectorize the docs in the corpus
		vect = TfidfVectorizer(min_df=1)
		tfidf = vect.fit_transform(docs)
		matrix = (tfidf * tfidf.T).A
		similarity_avg = sum(matrix[0]) / len(matrix[0])
		return similarity_avg

def compute():
	""" 
	Utility function that loads a corpus of agreements to populate db.classified
	"""
	print("obtain a corpus...")
	from classifier import build_corpus
	corpus = build_corpus()

	print("load the datastore...")
	datastore = WiserDatabase()
	cnt = 0
	for filename in corpus.fileids():
		cnt = cnt + 1
		if cnt > 1:
			print("\nend")
			return
		print("analyzing file %s..." % filename)
		record = datastore.fetch_by_filename(filename)
		print("record contains: ")
		print(record)

		if (record is not None):
			object_id = str(record['_id'])
			category = record['category']
			schema = AgreementSchema()
			schema.load_schema(category)
			aligner = Alignment(schema=schema)
			doc = corpus.raw(filename)
			paras = aligner.tokenize(doc)
			aligned_provisions = aligner.align(paras) # aligned_provisions is a list of tuples
			tupleized = aligner.continguous_normalize(aligned_provisions)
			print([a[0] for a in tupleized])
			analysis = AgreementStatistics(tupleized=tupleized, raw=doc)
			stats = analysis.calculate_stats()
			doc_gulpease = analysis.calculate_complexity(doc)
			stats['doc_gulpease2'] = doc_gulpease
			stats['category'] = category
			print("update the datastore")
			result = datastore.update_record(filename=filename, parameters=stats)

def compute_contract_group_info():
	""" 
	Utility function that loads a corpus of agreements to populate db.classified
	"""
	print("load the datastore...")
	from helper import WiserDatabase
	datastore = WiserDatabase()

	print("obtain a corpus...")
	from statistics import CorpusStatistics
	from classifier import build_corpus
	corpus = build_corpus()
	corpus_stats = CorpusStatistics(corpus)

	for category in corpus.categories():
		print("analyzing category %s..." % category)
		record = datastore.get_contract_group(category)

		stats = {}
		stats['group-similarity-score'] = corpus_stats.calculate_similarity(category=category)
		stats['group-complexity-score'] = corpus_stats.calculate_complexity(category=category)

		result = datastore.update_contract_group(agreement_type=category, info=stats)
		if (result.acknowledged and result.modified_count):
			print("matched count is %s" % str(result.matched_count))
			print("modified count is %s" % str(result.modified_count))

def display_contract_group_info():
	""" 
	prints out the contract_group collection for some metastats about contracts.
	"""
	print("load the datastore...")
	from helper import WiserDatabase
	datastore = WiserDatabase()

	print("obtain a corpus...")
	from classifier import build_corpus
	corpus = build_corpus()

	for category in corpus.categories():
		record = datastore.get_contract_group(category)
		print(record)

def compute_provision_group_info():
	""" 
	Utility function that loads a corpus of trainers to populate db.provision_group
	"""
	print("load the datastore...")
	from helper import WiserDatabase
	datastore = WiserDatabase()

	print("obtain a corpus...")
	from statistics import ProvisionStatistics
	provision_stats = ProvisionStatistics()
	provisions = provision_stats.provisions
	for (provision_name, fileid) in provisions:
		similarity_scores = []
		complexity_scores = []
		sents = provision_stats.corpus.sents(fileid)
		joined_sents = []
		for sent in sents:
			joined_sents.append(" ".join(sent))
		stats = {}
		stats['prov-similarity-avg'] = provision_stats.calculate_similarity(joined_sents)
		stats['prov-complexity-avg'] = provision_stats.calculate_complexity()
		result = datastore.update_provision_group(provision_name, stats)
		if (result.acknowledged and result.modified_count):
			print("matched count is %s" % str(result.matched_count))
			print("modified count is %s" % str(result.modified_count))

def display_provision_group_info():
	""" 
	prints out the provision_group collection for some metastats about contracts.
	"""
	print("load the datastore...")
	from helper import WiserDatabase
	datastore = WiserDatabase()
	# Get all the provision types 
	results = datastore.provision_group.find({})
	for provision_info in results:
		print(provision_info)

