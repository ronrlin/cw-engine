#!/usr/bin/python
from __future__ import division

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus.reader.plaintext import PlaintextCorpusReader

from helper import WiserDatabase
from structure import AgreementSchema
from structure import load_training_data
from alignment import Alignment

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
		tfidf = vect.fit_transform(provisions)

		from sklearn.metrics.pairwise import linear_kernel
		cosine_similarities = linear_kernel(tfidf[0:1], tfidf).flatten()
		stats = {}
		# TODO: watch out for self-similarity... exclude values of 1.0
		stats['mean'] = np.mean(cosine_similarities) * 100
		stats['median'] = np.mean(cosine_similarities) * 100
		stats['var'] = np.var(cosine_similarities) * 100
		stats['max'] = np.max(cosine_similarities) * 100
		stats['min'] = np.min(cosine_similarities) * 100
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

		char_cnt = [len(thistext) for thistext in text_blocks]
		word_cnt = [len(word_tokenize(thistext)) for thistext in text_blocks]
		values = [calculate_gulpease(thistext) for thistext in text_blocks]
		syllable_cnt = [count_syllables_in_text(thistext) for thistext in text_blocks]

		stats = {}
		stats['mean'] = np.mean(values)
		stats['median'] = np.mean(values)
		stats['var'] = np.var(values)
		stats['max'] = np.max(values)
		stats['min'] = np.min(values)

		stats['mean_length'] = np.mean(word_cnt)
		stats['median_length'] = np.median(word_cnt)
		stats['var_length'] = np.var(word_cnt)
		stats['max_length'] = np.max(word_cnt)
		stats['min_length'] = np.min(word_cnt)

		stats['mean_syllable'] = np.mean(syllable_cnt)
		stats['median_syllable'] = np.median(syllable_cnt)
		stats['var_syllable'] = np.var(syllable_cnt)
		stats['max_syllable'] = np.max(syllable_cnt)
		stats['min_syllable'] = np.min(syllable_cnt)

		print(stats)
		return stats

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
		#similarity_avg = (sum(matrix[0]) / len(matrix[0])) * 100
		stats = {}
		stats['mean'] = np.mean(matrix) * 100
		stats['median'] = np.median(matrix) * 100
		stats['var'] = np.var(matrix) * 100
		stats['max'] = np.max(matrix[0][matrix[0] != 1]) * 100 
		stats['min'] = np.min(matrix) * 100
		#return round(similarity_avg, 1)
		return stats

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

	def calculate_complexity(self, category=None):
		"""
		Calculates a global doc-to-doc complexity average for a corpus of interest.
		complexity may be computed relative to reference subset of type 'category'.

		:params category: string that represents a category

		Returns the average complexity across documents(category).
		"""
		docs = [self.corpus.raw(fileid) for fileid in self.corpus.fileids(category)]

		char_cnt = [len(text) for text in docs]
		word_cnt = [len(word_tokenize(text)) for text in docs]
		values = [calculate_gulpease(text) for text in docs]
		syllable_cnt = [count_syllables_in_text(text) for text in docs]

		stats = {}
		stats['mean'] = np.mean(values)
		stats['median'] = np.median(values)
		stats['var'] = np.var(values)
		stats['max'] = np.max(values)
		stats['min'] = np.min(values)		

		stats['length_mean'] = np.mean(word_cnt)
		stats['length_median'] = np.median(word_cnt)
		stats['length_var'] = np.var(word_cnt)
		stats['length_max'] = np.max(word_cnt)
		stats['length_min'] = np.min(word_cnt)		

		stats['syllable_mean'] = np.mean(syllable_cnt)
		stats['syllable_median'] = np.median(syllable_cnt)
		stats['syllable_var'] = np.var(syllable_cnt)
		stats['syllable_max'] = np.max(syllable_cnt)
		stats['syllable_min'] = np.min(syllable_cnt)		

		return stats

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

	def get_consensus(self, corpus, provision_type):
		""" Finds the incidence of provision_type in the corpus provided.    
		"""
		#print("get_consensus for %s" % provision_type)
		if corpus and len(corpus.fileids()) > 0:
			#print("there are %s files in the corpus" % str(len(corpus.fileids())))
			cnt = 0
			for fileid in corpus.fileids():
				search_info = { "filename" : fileid, "has_" + provision_type.replace("train/train_", "") : True}
				datastore = WiserDatabase()
				fs = datastore.fetch_by_tag(search_info)
				if len(fs):
					cnt = cnt + 1
			return round(100 * (cnt / len(corpus.fileids())), 0)
		else:
			return -1

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
			if _type:
				parameters["has_" + _type.replace("train/train_", "")] = True
			parameters[_type.replace("train/train_", "") + "_gulpease"] = self.calculate_complexity(_block)

		self.output = parameters
		return parameters

	def calculate_flesch(self, text):
		pass

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
		return round(gulpease, 1)

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
		similarity_avg = (sum(matrix[0]) / len(matrix[0])) * 100
		return round(similarity_avg, 1)

def calculate_gulpease(text):
	character_count = len(text)
	word_count = len(word_tokenize(text))
	sent_count = len(sent_tokenize(text))
	gulpease = 89 - 10 * (character_count/word_count) + 300 * (sent_count/word_count)
	return gulpease

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

def compute_classified_stats():
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
		print("analyzing file %s..." % filename)
		record = datastore.fetch_by_filename(filename)
		print("record contains: ")
		print(record)

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
			doc_gulpease = analysis.calculate_complexity(doc)
			stats['doc_gulpease2'] = doc_gulpease
			stats['category'] = category

			for (_block, _type) in aligned_provisions:
				if _type: 
					if ("train/train_" in _type):
						stats['has_' + _type.replace("train/train_", "")] = True
					else:
						stats['has_' + _type] = True

					#NER here?

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
		#record = datastore.get_contract_group(category)

		simstats = corpus_stats.calculate_similarity(category=category)
		stats = {}
		stats['group-similarity-score'] = simstats['mean']
		stats['group-similarity-var'] = simstats['var']
		stats['group-similarity-max'] = simstats['max']
		stats['group-similarity-min'] = simstats['min']

		comstats = corpus_stats.calculate_complexity(category=category)
		stats['group-complexity-score'] = comstats['mean']
		stats['group-complexity-var'] = comstats['var']
		stats['group-complexity-max'] = comstats['max']
		stats['group-complexity-min'] = comstats['min']
		# some additional detail
		stats['group-length-mean'] = comstats['length_mean']
		stats['group-length-var'] = comstats['length_var']
		stats['group-length-max'] = comstats['length_max']
		stats['group-length-min'] = comstats['length_min']
		stats['group-syllabus-mean'] = comstats['syllable_mean']
		stats['group-syllabus-var'] = comstats['syllable_var']
		stats['group-syllabus-max'] = comstats['syllable_max']
		stats['group-syllabus-min'] = comstats['syllable_min']

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

def compute_curated_info():
	""" 
	This function will go through the db.curated in mongo and calculate
	stats about the text provisions.  
	"""
	print("load the datastore...")
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
	print("load the datastore...")
	from helper import WiserDatabase
	datastore = WiserDatabase()

	# obtain a list of provision text, type

	for (name, path) in load_training_data().items():
		print("loading provision %s " % name)
		stats = ProvisionStatistics(name)
		info = {}
		siminfo = stats.calculate_similarity()
		info['prov-similarity-avg'] = siminfo['mean']
		info['prov-similarity-median'] = siminfo['median']
		info['prov-similarity-var'] = siminfo['var']
		info['prov-similarity-max'] = siminfo['max']
		info['prov-similarity-min'] = siminfo['min']
		cominfo = stats.calculate_complexity()
		info['prov-complexity-avg'] = cominfo['mean'] 
		info['prov-complexity-median'] = cominfo['median'] 
		info['prov-complexity-var'] = cominfo['var'] 
		info['prov-complexity-max'] = cominfo['max'] 
		info['prov-complexity-min'] = cominfo['min'] 
		info['prov-length-avg'] = cominfo['mean_length']
		info['prov-length-median'] = cominfo['median_length'] 
		info['prov-length-var'] = cominfo['var_length'] 
		info['prov-length-max'] = cominfo['max_length'] 
		info['prov-length-min'] = cominfo['min_length']
		info['prov-syllable-avg'] = cominfo['mean_syllable']
		info['prov-syllable-median'] = cominfo['median_syllable'] 
		info['prov-syllable-var'] = cominfo['var_syllable'] 
		info['prov-syllable-max'] = cominfo['max_syllable'] 
		info['prov-syllable-min'] = cominfo['min_syllable']

		result = datastore.update_provision_group(name, info)
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

def testing_calcs():
	sample_provision = "Some text that represents a provision from an uploaded agreement."

	# load ProvisionStats with no provision specified.
	stats = ProvisionStatistics("confidential_information")
	comp_score = stats.calculate_complexity(text=sample_provision)
	sim_score = stats.calculate_similarity(text=sample_provision)
	print("comp_score is %s" % str(comp_score))
	print("sim_score is %s" % str(sim_score))
	#astats = AgreementStatistics(tupleized)
	#result = stats.calculate_complexity()
	#result = stats.calculate_similarity()