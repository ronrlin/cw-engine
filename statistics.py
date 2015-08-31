#!/usr/bin/python
from helper import WiserDatabase
from structure import AgreementSchema
from alignment import Alignment
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from scipy.spatial.distance import cosine

from helper import WiserDatabase
from structure import AgreementSchema
from alignment import Alignment

class AgreementStatistics(object):
	""" """
	def __init__(self, tupleized, raw=None):
		""" """
		self.tupleized = tupleized
		self.raw = raw

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

		return parameters

	def calculate_complexity(self, text):
		""" 
		Function that computes complexity.
		:param text: is a string 
		Returns a score from 0 to 100
		"""
		character_count = len(text)
		word_count = len(word_tokenize(text))
		sent_count = len(sent_tokenize(text))
		gulpease = 89 - 10 * (character_count/word_count) + 300 * (sent_count/word_count)
		return gulpease

	def calculate_similarity(self, a, b):
		print("calculate cosine similarity...")
		similarity=cosine(a, b)
		print(similarity)
		return similarity

# Need an algorithm to collapse contiguous blocks!!  argghhh

def compute():
	""" """
	print("obtain a corpus...")
	from classifier import get_agreement_corpus
	corpus = get_agreement_corpus()
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


def helpme():
	filename = "nda-0000-0014.txt"
	print("obtain a corpus...")
	from classifier import get_agreement_corpus
	corpus = get_agreement_corpus()
	print("load the datastore...")
	datastore = WiserDatabase()
	print("analyzing file %s..." % filename)
	record = datastore.fetch_by_filename(filename)

	schema = AgreementSchema()
	schema.load_schema(record['category'])
	aligner = Alignment(schema=schema)
	doc = corpus.raw(filename)
	paras = aligner.tokenize(doc)
	aligned_provisions = aligner.align(paras) # aligned_provisions is a list of tuples
	tupleized = aligner.continguous_normalize(aligned_provisions)
	print(len(tupleized))
	print(tupleized)
	#return aligned_provisions

	newdoc = [block for (block, type) in tupleized]
	aligned_provisions = aligner.align(newdoc)
	print("--------")
	print(len(aligned_provisions))
	print(aligned_provisions)

	#analysis = AgreementStatistics(tupleized=tupleized, raw=corpus.raw(filename))
	#stats = analysis.calculate_stats()
	#doc_gulpease = analysis.calculate_complexity(corpus.raw(filename))

