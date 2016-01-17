#!/usr/bin/python
from __future__ import division

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus.reader.plaintext import PlaintextCorpusReader

from structure import AgreementSchema
from structure import load_training_data
from alignment import Alignment

import os
import numpy as np
from nltk.tokenize import BlanklineTokenizer

"""
CorpusStatistics(corpus)
AgreementStatistics(tupleized)
ProvisionStatistics()

The idea is to create instances of the objects above, which write data to the datastore.

function compute(...)
"""
BASE_PATH = "./"
DATA_PATH = os.path.join(BASE_PATH, "data/")

class ProvisionStatistics(object):
	""" 
	It would be nice to ask this object:

	compare this text to text that is identified as a certain provision
	p = ProvisionStatistics(category)
	p.calculate_similarity(provision_text)

	if a category is not specified, then load all.
	calculate_similarity(self, text) #calculate the similarity to the reference corpus
	if text=None, then just return the average of all.
	if text=something, then compare the similarity of the text to the reference

	calculate_complexity(self, text)
	if text=None, then calcualte the average
	if text=something then calculate the given text compared to reference

	TODO: this should load from the curated_db instead of trainers
	"""
	def __init__(self, provision_name=None):
		# load the train/train_* files into a corpus
		training = load_training_data().items()
		if provision_name is not None:			
			self.provisions = [(name,path) for (name, path) in training if name==provision_name]
		else:
			self.provisions = training
		# provisions is a tuple (provision_name, provision_path)
		training_file_names = [p[1] for p in self.provisions]
		self.provision_names = [p[0] for p in self.provisions]
		self.corpus = PlaintextCorpusReader(BASE_PATH, training_file_names)

	def calculate_similarity(self, text=None):
		""" 
		Calculate the similarity of the given text. If no text parameter 
		is specified then an average similarity score is calculated for
		all the tokenized sentences that are found in the texts of the corpus.    

		:param text: is a string 

		Returns a floating point value.
		"""		
		b = BlanklineTokenizer()
		rawtext = [b.tokenize(self.corpus.raw(fileid)) for fileid in self.corpus.fileids()]
		provisions = [paragraph for doc in rawtext for paragraph in doc]
		vect = TfidfVectorizer(min_df=1)
		if text is not None:
			provisions.insert(0, text)
		# TODO!!! look closely... i think i need to separate fit and transform here!
		tfidf = vect.fit_transform(provisions)

		from sklearn.metrics.pairwise import linear_kernel
		cosine_similarities = linear_kernel(tfidf[0:1], tfidf).flatten()
		stats = {}
		stats['similarity'] = {
			'mean' : np.mean(cosine_similarities) * 100,
			'median' : np.median(cosine_similarities) * 100,
			'var' : np.var(cosine_similarities) * 100,
			'max' :  np.max(cosine_similarities) * 100,
			'min' : np.min(cosine_similarities) * 100,
		}
		return stats

	def calculate_complexity(self, text=None):
		"""
		Calculate the complexity of the given text. If no text parameter 
		is specified, then the average complexity is calculated for the 
		corpus loaded.    

		:param text: is a string 

		Returns a floating point value.
		"""
		text_blocks = []
		if text is not None:
			text_blocks = [text]
		else:
			b = BlanklineTokenizer()
			text_blocks = [b.tokenize(self.corpus.raw(fileid)) for fileid in self.corpus.fileids()]
			text_blocks = [paragraph for doc in text_blocks for paragraph in doc]

		stats = {}

		values = [calculate_gulpease(thistext) for thistext in text_blocks]
		stats['gulpease'] = {
			'mean' : np.mean(values),
			'median' : np.mean(values),
			'var' : np.var(values),
			'max' : np.max(values),
			'min' : np.min(values),
		}
		values = [calculate_flesch(thistext) for thistext in text_blocks]
		stats['flesch'] = {
			'mean' : np.mean(values),
			'median' : np.mean(values),
			'var' : np.var(values),
			'max' : np.max(values),
			'min' : np.min(values),
		}
		word_cnt = [len(word_tokenize(thistext)) for thistext in text_blocks]
		stats['word_count'] = {
			'mean' : np.mean(word_cnt),
			'median' : np.median(word_cnt),
			'var' : np.var(word_cnt),
			'max' : np.max(word_cnt),
			'min' : np.min(word_cnt),
		}
		syllable_cnt = [count_syllables_in_text(thistext) for thistext in text_blocks]		
		stats['syllable_count'] = {
			'mean' : np.mean(syllable_cnt),
			'median' : np.median(syllable_cnt),
			'var' : np.var(syllable_cnt),
			'max' : np.max(syllable_cnt),
			'min' : np.min(syllable_cnt),
		}
		char_cnt = [len(thistext) for thistext in text_blocks]
		stats['char_count'] = {
			'mean' : np.mean(char_cnt),
			'median' : np.median(char_cnt),
			'var' : np.var(char_cnt),
			'max' : np.max(char_cnt),
			'min' : np.min(char_cnt),
		}
		#print(stats)
		return stats

class CorpusStatistics(object):
	""" """
	def __init__(self, corpus):
		self.corpus = corpus

	def transform(self):
		"""

					gulpease 	syllables 	words	
		gulpease 	   34 			31		 21


		mean		gulpease 	syllables 	words	
		gulpease	22.3 			31 		 22


		var			preamble 	recital		stuff


		"""


	def calculate_similarity(self, text=None):
		""" 
		Calculates a global doc-to-doc similarity average for a corpus of interest.
		Similarity may be computed relative to reference subset of type 'category'.

		:params category: string that represents a category

		Returns the average similarity across documents.
		"""
		docs = [self.corpus.raw(fileid) for fileid in self.corpus.fileids()]
		vect = TfidfVectorizer(min_df=1)
		vect.fit(docs)

		if text is not None:
			docs.insert(0, text)
		
		tfidf = vect.fit_transform(docs)

		from sklearn.metrics.pairwise import linear_kernel
		cosine_similarities = linear_kernel(tfidf[0:1], tfidf).flatten()

		newstats = {}
		newstats['similarity'] = {
			'mean' : np.mean(cosine_similarities) * 100,
			'median' : np.median(cosine_similarities) * 100,
			'var' : np.var(cosine_similarities) * 100,
			'max' : np.max(cosine_similarities) * 100,
			'min' : np.min(cosine_similarities) * 100,		
		}
		return newstats

	def most_similar(self, doc, limit=5):
		""" Determines what files are most similar to the presented doc. """
		docs = [self.corpus.raw(fileid) for fileid in self.corpus.fileids()]
		docs.insert(0, doc)
		vect = TfidfVectorizer(min_df=1)
		tfidf = vect.fit_transform(docs)
		from sklearn.metrics.pairwise import linear_kernel
		cosine_similarities = linear_kernel(tfidf[0:1], tfidf).flatten()
		neg_idx = limit * -1
		related_docs_indices = cosine_similarities.argsort()[:neg_idx:-1]

		fileids = self.corpus.fileids()
		fileids.insert(0, "newone")
		fileids = np.array(fileids)

		return(fileids[related_docs_indices[1:]])

	def calculate_complexity(self):
		"""
		Calculates a global doc-to-doc complexity average for a corpus of interest.
		complexity may be computed relative to reference subset of type 'category'.

		:params category: string that represents a category

		Returns the average complexity across documents(category).
		"""
		docs = [self.corpus.raw(fileid) for fileid in self.corpus.fileids()]

		newstats = {}

		values = [calculate_gulpease(text) for text in docs]
		newstats['gulpease'] = {
			'mean' : np.mean(values),
			'median' : np.median(values),
			'var' : np.var(values),
			'max' : np.max(values),
			'min' : np.min(values),		
		}
		flesch = [calculate_flesch(text) for text in docs]		
		newstats['flesch'] = {
			'mean' : np.mean(flesch),
			'median' : np.median(flesch),
			'var' : np.var(flesch),
			'max' : np.max(flesch),
			'min' : np.min(flesch),		
		}
		word_cnt = [len(word_tokenize(text)) for text in docs]		
		newstats['word_count'] = {
			'mean' : np.mean(word_cnt),
			'median' : np.median(word_cnt),
			'var' : np.var(word_cnt),
			'max' : np.max(word_cnt),
			'min' : np.min(word_cnt),		
		}
		char_cnt = [len(text) for text in docs]		
		newstats['char_count'] = {
			'mean' : np.mean(char_cnt),
			'median' : np.median(char_cnt),
			'var' : np.var(char_cnt),
			'max' : np.max(char_cnt),
			'min' : np.min(char_cnt),		
		}
		syllable_cnt = [count_syllables_in_text(text) for text in docs]
		newstats['syllable_count'] = {
			'mean' : np.mean(syllable_cnt),
			'median' : np.median(syllable_cnt),
			'var' : np.var(syllable_cnt),
			'max' : np.max(syllable_cnt),
			'min' : np.min(syllable_cnt),		
		}

		return newstats

class AgreementStatistics(object):
	""" """
	def __init__(self, tupleized, raw=None):
		""" """
		self.tupleized = tupleized

	def transform(self):
		""" Transform tupleized into numpy arrays of numbers to do some statistics. """
		feature_names = [_type.replace("train/train_", "") for _block, _type in self.tupleized]
		self._feature_names = list(set(feature_names))
		print("_feature_names")
		print(self._feature_names)
		newtup= []
		for f in self._feature_names:
			blocks = " ".join([_block for _block, _type in self.tupleized if f in _type])
			newtup.append((blocks, f))
		self.gulpease = np.array([calculate_gulpease(_block) for _block, _type in newtup])
		self.flesch = np.array([calculate_flesch(_block) for _block, _type in newtup])
		self.syllables = np.array([count_syllables_in_text(_block) for _block, _type in newtup])
		self.words = np.array([len(word_tokenize(_block)) for _block, _type in newtup])
		return

	#def anamolous(self):
	#	value_matrix = [gulpease, syllables, words]
	#	datastore = WiserDatabase()
	#	gulpease_mean = np.array([datastore.get_provision_group(feature)['prov-complexity-avg'] for feature in feature_names])
	#	gulpease_var = np.array([datastore.get_provision_group(feature)['prov-complexity-var'] for feature in feature_names])
	#	gulpease_max = np.array([datastore.get_provision_group(feature)['prov-complexity-max'] for feature in feature_names])
	#	gulpease_deviations = (gulpease - gulpease_mean) / np.sqrt(gulpease_var)
	#	print("gulpease_deviations")
	#	print(gulpease_deviations)
	#	syllable_mean = [datastore.get_provision_group(feature)['prov-syllable-avg'] for feature in feature_names]
	#	print("syllable_mean")
	#	print(syllable_mean)
	#	syllable_var = [datastore.get_provision_group(feature)['prov-syllable-var'] for feature in feature_names]
	# print("syllable_var")
	# print(syllable_var)
	# syllable_max = [datastore.get_provision_group(feature)['prov-syllable-max'] for feature in feature_names]
	# syllable_deviations = (np.array(syllables) - np.array(syllable_mean)) / np.sqrt(np.array(syllable_var))
	# print("syllable_deviations")
	# print(syllable_deviations)
	# word_mean = np.array([datastore.get_provision_group(feature)['prov-length-avg'] for feature in feature_names])
	# print("word_mean")
	# print(word_mean)
	# word_var = np.array([datastore.get_provision_group(feature)['prov-length-var'] for feature in feature_names])
	# print("word_var")
	# print(word_var)
	# word_max = np.array([datastore.get_provision_group(feature)['prov-length-max'] for feature in feature_names])
	# word_deviations = (words - word_mean) / np.sqrt(word_var)
	# print("word_deviations")
	# print(word_deviations)

	# self._feature_names = feature_names
	# self._transformed = np.array([gulpease_deviations, syllable_deviations, word_deviations])
	# names = np.array(feature_names)
	# print(names[word_deviations > 2])
		#matrix.mean(axis=0)

	def get_gulpease(self, _type):
		col_index = self._feature_names.index(_type)
		return self.gulpease[col_index]

	def get_flesch(self, _type):
		col_index = self._feature_names.index(_type)
		return self.flesch[col_index]

	def get_syllable_count(self, _type):
		col_index = self._feature_names.index(_type)
		return self.syllables[col_index]

	def get_word_count(self, _type):
		col_index = self._feature_names.index(_type)
		return self.words[col_index]

	#def get_output(self, feature_name):
	#	""" 
	#	This function builds the document that will be kept in the proper datastore.
	#	Returns a dict that represents statistics about an agreement.
	#	"""
	#	col_index = self._feature_names.index(feature_name)
	#	deviations = self._transformed[:,col_index]
	#	return list(deviations)

	def calculate_stats(self):
		""" """
		doc = " ".join([e[0] for e in self.tupleized])
		parameters = {}
		parameters['doc_gulpease'] = calculate_gulpease(doc)
		parameters['doc_flesch'] = calculate_flesch(doc)
		parameters["word_count"] = len(word_tokenize(doc))
		parameters["para_count"] = len(self.tupleized)
		parameters["syllable_count"] = count_syllables_in_text(doc)
		for (_block, _type) in self.tupleized:
			if _type:
				parameters["has_" + _type.replace("train/train_", "")] = True
				parameters[_type.replace("train/train_", "") + "_gulpease"] = calculate_gulpease(_block)
				parameters[_type.replace("train/train_", "") + "_flesch"] = calculate_flesch(_block)
		return parameters

	#def calculate_complexity(self, text):
	#	""" 
	#	Function that computes the Gulpease score.
	#	:param text: string 
	#	"""
	#	return calculate_gulpease(text)

	#def calculate_similarity(self, text, corpus):
	#	""" 
	#	This SHOULD compare a given text to a reference corpus.
	#	:param text: string of text
	#	:param b: string of text
	#	"""
	#	docs = [corpus.raw(fileid) for fileid in corpus.fileids()]
	#	docs.append(text)
	#	# vectorize the docs in the corpus
	#	vect = TfidfVectorizer(min_df=1)
	#	tfidf = vect.fit_transform(docs)
	#	matrix = (tfidf * tfidf.T).A
	#	similarity_avg = (sum(matrix[0]) / len(matrix[0])) * 100
	#	return round(similarity_avg, 1)

def calculate_gulpease(text):
	character_count = len(text)
	word_count = len(word_tokenize(text))
	sent_count = len(sent_tokenize(text))
	if not word_count:
		return 1
	else: 
		gulpease = 89 - 10 * (character_count/word_count) + 300 * (sent_count/word_count)
		return gulpease

def calculate_flesch(text):
	""" Calculate the Flesch Reading Ease score, yielding values between 0 and 120. """
	word_count = len(word_tokenize(text))
	sent_count = len(sent_tokenize(text))
	syllable_count = count_syllables_in_text(text)
	flesch_score = 206.835 - 1.015* (word_count/sent_count) - 84.6 * (syllable_count / word_count)
	return flesch_score

def count_syllables_in_text(text):
	words = word_tokenize(text)
	syllable_cnt = []
	for word in words:
		syllable_cnt.append(count_syllables_in_word(word))
	return np.sum(syllable_cnt)

def count_syllables_in_word(word):
	vowels = ['a','e','i','o','u','y']
	count = 0
	for i in range(0,len(word)):
		if(word[i] in vowels and (i+1>=len(word) or word[i+1] not in vowels)):
			count+=1
	return count

def get_consensus(corpus, provision_type):
	""" Finds the incidence of provision_type in the corpus provided. """
	from helper import WiserDatabase
	datastore = WiserDatabase()
	if corpus and len(corpus.fileids()) > 0:
		cnt = 0
		for fileid in corpus.fileids():
			search_info = { "filename" : fileid, "has_" + provision_type.replace("train/train_", "") : True}
			fs = datastore.fetch_by_tag(search_info)
			if len(fs):
				cnt = cnt + 1
		return round(100 * (cnt / len(corpus.fileids())), 0)
	else:
		# TODO: error condition or assert _something_
		return 0

def compute_classified_stats():
	""" 
	Utility function that loads a corpus of agreements to populate db.classified
	"""
	from classifier import build_corpus
	corpus = build_corpus()

	from helper import WiserDatabase
	datastore = WiserDatabase()
	cnt = 0
	# TODO!!!! you should use curated db for this 
	for filename in corpus.fileids():
		record = datastore.fetch_by_filename(filename)
		if (record is not None):
			object_id = str(record['_id'])
			category = record['category']
			schema = AgreementSchema()
			schema.load_schema(category)
			aligner = Alignment(schema=schema, vectorizer=2, all=True)
			doc = corpus.raw(filename)
			paras = aligner.tokenize(doc)
			aligned_provisions = aligner.align(paras, version=2) # aligned_provisions is a list of tuples
			#aligned_provisions = aligner.sanity_check(aligned_provisions)

			analysis = AgreementStatistics(tupleized=aligned_provisions)
			stats = analysis.calculate_stats()
			stats['category'] = category

			for (_block, _type) in aligned_provisions:
				if _type: 
					stats['has_' + _type.replace("train/train_", "")] = True

			result = datastore.update_record(filename=filename, parameters=stats)
	print("compute_classified_stats completed.")

def compute_contract_group_info():
	""" 
	Utility function that loads a corpus of agreements to populate db.classified
	"""
	from helper import WiserDatabase
	datastore = WiserDatabase()

	from statistics import CorpusStatistics
	for category in datastore.get_category_names():
		records = datastore.fetch_by_category(category)
		fileids = [r['filename'] for r in records]
		corpus = PlaintextCorpusReader(DATA_PATH, fileids)
		corpus_stats = CorpusStatistics(corpus)
		stats = corpus_stats.calculate_similarity()
		comstats = corpus_stats.calculate_complexity()
		stats.update(comstats)
		result = datastore.update_contract_group(agreement_type=category, info=stats)
		if (result.acknowledged and result.modified_count):
			print("matched count is %s" % str(result.matched_count))
			print("modified count is %s" % str(result.modified_count))

def display_contract_group_info():
	""" 
	prints out the contract_group collection for some metastats about contracts.
	"""
	from helper import WiserDatabase
	datastore = WiserDatabase()

	from classifier import build_corpus
	corpus = build_corpus()

	for category in corpus.categories():
		record = datastore.get_contract_group(category)
		print(record)

def compute_curated_info():
	""" 
	This function will go through the db.curated in mongo and calculate
	stats about the text provisions.  
	"""
	from helper import WiserDatabase
	datastore = WiserDatabase()
	all_types = datastore.fetch_provision_types()
	for provision_type in all_types:
		provisions = datastore.fetch_provision_type(provision_type)
		for provision in provisions:
			stats = {}
			stats['gulpease'] = calculate_gulpease(provision['text'])
			stats['syllables'] = count_syllables_in_text(provision['text'])
			datastore.update_curated(provision['_id'], stats)

def compute_provision_group_info():
	""" 
	This function calculates provision_group info and writes it to the MongoDB.
	It loads all training files and calculates similarity/complexity
	for each sentence in the training file.  
	"""
	from helper import WiserDatabase
	datastore = WiserDatabase()
	for (name, path) in load_training_data().items():
		print("loading provision %s " % name)
		stats = ProvisionStatistics(name)
		siminfo = stats.calculate_similarity()
		cominfo = stats.calculate_complexity()
		siminfo.update(cominfo)
		result = datastore.update_provision_group(name, siminfo)
		if (result.acknowledged and result.modified_count):
			print("matched count is %s" % str(result.matched_count))
			print("modified count is %s" % str(result.modified_count))

def display_provision_group_info():
	""" 
	prints out the provision_group collection for some metastats about contracts.
	"""
	from helper import WiserDatabase
	datastore = WiserDatabase()
	results = datastore.provision_group.find({})
	for provision_info in results:
		print(provision_info)
