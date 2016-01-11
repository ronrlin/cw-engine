#!/usr/bin/python
import os
from sklearn import svm
from nltk.corpus.reader.plaintext import CategorizedPlaintextCorpusReader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

BASE_PATH = "./"
CORPUS_PATH = os.path.join(BASE_PATH, "data/")
DUMP_PATH = os.path.join(BASE_PATH, "dump/")

class AgreementVectorClassifier(object):
	""" """
	def __init__(self, vectorizer, corpus):
		""" """
		self.corpus = corpus
		self.vectorizer = vectorizer
		self.classifier = None

	def fit(self):
		"""
		This function fits the vectorizer and then trains the classifier.
		"""
		fileids = list(self.corpus._map.keys())
		cats = list(self.corpus._map.values())
		textcomp2 = [self.corpus.raw(fileid) for fileid in fileids]
		train_vector = self.vectorizer.fit_transform(textcomp2)
		self.classifier = svm.LinearSVC(class_weight='auto')
		self.classifier.fit(train_vector, cats)

	def classify_file(self, filename):
		fh = open("data/" + filename, 'r')
		x = fh.read()
		fh.close()
		dtm_test = self.vectorizer.transform([x])
		results = self.classifier.predict(dtm_test)
		return results[0]

	def classify_data(self, data):
		"""
		Classifies a body of text, represented as a list of strings. 
	
		:param data: should be a list of strings.
			ie: data = [ "string1", "string2", "string3" .. ]
		"""
		#print(data)
		data_vectorized = self.vectorizer.transform(data)
		results = self.classifier.predict(data_vectorized)
		# TODO: maybe log the size the result vector?
		#print("results!")
		#print(results)
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

class AgreementGaussianNBClassifier(object):
	""" Will use GaussianNB from the sklearn package """
	def __init__(self):
		pass

	def fit(self):
		pass

	def predict(self):
		pass

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
		# Note: don't make them nested!
		# It was unclear why this was the case, but the categories 
		# are not loaded correctly when you don't provide the cats as a 
		# parameter nested in a list.
		#newvals[f] = corpus_details[f]
		#binvals[f] = cats[i]
		newvals[f] = [corpus_details[f]]
		binvals[f] = [cats[i]]
		i = i + 1

	print("%s files loaded into corpus." % str(len(fileids)))
	# newvals and binvals is a dict, with a key to a list.  Therefore, k['key'] = ['category']
	if (binary):
		train_corpus = CategorizedPlaintextCorpusReader(CORPUS_PATH, fileids, cat_map=binvals)
	else:
		train_corpus = CategorizedPlaintextCorpusReader(CORPUS_PATH, fileids, cat_map=newvals)

	return train_corpus

def get_agreement_classifier_v1(train_corpus):
	""" """
	print("loaded CountVectorizer...")
	count_vectorizer = CountVectorizer(input='content', stop_words=None, ngram_range=(1,2))
	countClass = AgreementVectorClassifier(count_vectorizer, train_corpus)
	countClass.fit()
	return countClass

def get_agreement_classifier_v2(train_corpus):
	""" """
	print("loaded TfidfVectorizer...")
	tfidf_vectorizer = TfidfVectorizer(input='content', stop_words=None, ngram_range=(1,2))	
	tfidfClass = AgreementVectorClassifier(tfidf_vectorizer, train_corpus)
	tfidfClass.fit()
	return tfidfClass

def get_agreement_classifier_v3(train_corpus):
	""" """
	print("loaded NaiveBayesian, not stable.")
	naiveClass = AgreementNaiveBayesClassifier(train_corpus)
	naiveClass.fit()
	return naiveClass

#maybe move these to a curate.py
def scroll_through():
	from helper import WiserDatabase
	from alignment import Alignment
	from structure import AgreementSchema

	datastore = WiserDatabase()
	records = datastore.fetch_by_category("nondisclosure")
	cnt = 0
	schema = AgreementSchema()
	schema.load_schema("nondisclosure")
	aligner = Alignment(schema=schema, vectorizer=2, all=True)

	for record in records: 
		print(record['filename'])
		func(aligner, record['filename'])
		if cnt > 7:
			return
		cnt = cnt + 1

#maybe move these to a curate.py
# pre-curation functions
def func(aligner, filename="nda-0000-0008.txt"):
	import json
	import ner
	import config
	from feature import Feature
	import nltk

	ner_settings = config.load_ner()
	tagger = ner.SocketNER(host=ner_settings['hostname'], port=int(ner_settings['port']))

	print("Loading the agreement corpus...")
	corpus = build_corpus(binary=False)
	print("Loading the agreement classifier...")
	#classifier = get_agreement_classifier_v1(corpus)
	print("Application ready to load.")

	doc = corpus.raw(filename)

	#aligner = Alignment(schema=schema, vectorizer=2, all=True)
	paras = aligner.tokenize(doc)
	paras = aligner.simplify(paras)
	aligned_provisions = aligner.align(paras, version=2)

	feature = Feature()
	provision_features = feature.text_identify(paras)

	"""
	{
		filename : "",
		document_class : "nondisclosure"
		bag_of_words,
		first_guess : "time_period",
		feature_class : "normal_text",
		paragraph_index : 3,
		relative_paragraph_index : 3 / paragraph_count,
		prev_guess : "severability",
		next_guess : "severability",
		prev_actual : "severability",
		next_actual : "severability",
		containsDATE : true,
		containsMONEY : false,
		containsORGANIZATION : true,
		containsPERSON : true,
		paragraph_complexity : 45.2,
		paragraph_similarity : 49.0,

		confirm_class : "severability",
	}
	"""
	pidx = 0
	provision_count = len(aligned_provisions)
	allinfo = []
	for proviso in aligned_provisions:
		print(proviso)
		property_dict = dict()
		property_dict['filename'] = filename
		property_dict['document_class'] = "nondisclosure"
		property_dict['first_guess'] = proviso[1]
		property_dict['paragraph_index'] = pidx
		property_dict['relative_paragraph_index'] = round(float(100) * (float(pidx) / provision_count), 2)
		property_dict['feature_class'] = provision_features[pidx][1]
		property_dict['character_count'] = len(proviso[0])
		property_dict['text'] = proviso[0]

		words = nltk.tokenize.word_tokenize(proviso[0])
		property_dict['word_count'] = len(words)

		nerz = tagger.get_entities(proviso[0])
		types = ["DATE", "ORGANIZATION", "PERSON", "LOCATION","MONEY", "PERCENT", "TIME"]
		missing = set(types) - set(nerz.keys())
		for k in missing:
			property_dict["contains" + k] = False

		for k in nerz.keys():
			if nerz[k]:
				property_dict["contains" + k] = True

		if pidx > 0 and pidx < provision_count-1:
			property_dict['next_guess'] = aligned_provisions[pidx+1][1]
			property_dict['prev_guess'] = aligned_provisions[pidx-1][1]
		else:
			if pidx == 0:
				property_dict['next_guess'] = aligned_provisions[pidx+1][1]
				property_dict['prev_guess'] = None
			elif pidx == provision_count-1:
				property_dict['next_guess'] = None
				property_dict['prev_guess'] = aligned_provisions[pidx-1][1]

		allinfo.append(property_dict)
		pidx += 1

	with open("parsed-" + filename, 'w') as outfile:
		json.dump(allinfo, outfile, indent=4, sort_keys=True)

def get_text_from_file(filename, output):
	import config
	tika = config.load_tika()
	tika_url = "http://" + tika['hostname'] + ":" + tika['port'] + "/tika"

	import requests
	contract_data = ""
	if (".pdf" in filename):					
		print("filename: %s" % filename)
		r=requests.put(tika_url, data=output, headers={"Content-type" : "application/pdf", "Accept" : "text/plain"})
		contract_data = r.text.encode("ascii", "replace")
		contract_data = contract_data.replace("?", " ")

	elif (".docx" in filename):
		print("filename: %s" % filename)
		r=requests.put(tika_url, data=output, headers={"Content-type" : "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "Accept" : "text/plain"})
		contract_data = r.text.encode("ascii", "replace")
		contract_data = contract_data.replace("?", " ")

	elif (".doc" in filename):
		print("filename: %s" % filename)
		r=requests.put(tika_url, data=output, headers={"Content-type" : "application/msword", "Accept" : "text/plain"})
		contract_data = r.text.encode("ascii", "replace")
		contract_data = contract_data.replace("?", " ")

	elif (".rtf" in filename):
		print("filename: %s" % filename)
		r=requests.put(tika_url, data=output, headers={"Content-type" : "application/rtf", "Accept" : "text/plain"})
		contract_data = r.text.encode("ascii", "replace")
		contract_data = contract_data.replace("?", " ")

	else: # (".txt" in filename): 
		#print("not necessary to transform files that are already txt.")
		pass

	return contract_data

# BOOTSTRAP function
def transform_to_text():
	""" Iterates through all files in the CORPUS PATH, creates a TXT file for any file type other than a TXT file. 
		Once a file has been created, it moves the original file to the DUMP directory

		How to add files to the corpus:
		1. Add a file into the data/ path
		2. Add a record into the master-classifier.csv
		3. Run python:
			>> import classifier
			>> classifier.transform_to_text()
	"""
	for filename in os.listdir(CORPUS_PATH):
		
		with open(os.path.join(CORPUS_PATH, filename),'rb') as _file:
			output = _file.read()
		
		contract_data = get_text_from_file(filename, output)
		if contract_data:
			print("processing %s" % filename)
			newname = filename.replace(".pdf", ".txt")
			f = open(os.path.join(CORPUS_PATH, newname), "w")
			f.write(contract_data)
			f.close()
			print("created a new file, %s" % newname)

			import shutil
			shutil.move(os.path.join(CORPUS_PATH, filename), os.path.join(DUMP_PATH, filename))
			print("removed the original file")
	print("transformations completed.")

def testing():
	"""
	To use classifiers using vectorization, you can obtain a 
	document through the corpus using the raw(filename) function.
	"""
	print("Loading the datastores...")
	from helper import WiserDatabase
	datastore = WiserDatabase()
	print("Build a corpus of documents...")
	corpus = build_corpus(binary=False, binary_param='nondisclosure')
	print("Loading the agreement classifier...")
	classifier = get_agreement_classifier_v1(corpus)
	print("Application ready to load.")
	#agreement_type = classifier.classify_file("/home/obironkenobi/Projects/ContractWiser/small-data/nda-0000-0025.txt")
	"""
	a doc should be a list of strings : [ "string1", "string2", "string3" .. ]
	"""
	doc = corpus.raw("nda-0000-0008.txt")
	agreement_type = classifier.classify_data([doc])
	print("The nda agreement as list is a %s agreement" % agreement_type)

	doc = corpus.raw("nda-0000-0008.txt")
	agreement_type = classifier.classify_data([doc])
	print("The nda agreement as list is a %s agreement" % agreement_type)

	doc = corpus.raw("nda-0000-0009.txt")
	agreement_type = classifier.classify_data([doc])
	print("The nda agreement as list is a %s agreement" % agreement_type)

	doc = corpus.raw("nda-0000-0010.txt")
	agreement_type = classifier.classify_data([doc])
	print("The nda agreement as list is a %s agreement" % agreement_type)

	doc = corpus.raw("nda-0000-0011.txt")
	agreement_type = classifier.classify_data([doc])
	print("The nda agreement as list is a %s agreement" % agreement_type)

	doc = corpus.raw("nda-0000-0012.txt")
	agreement_type = classifier.classify_data([doc])
	print("The nda agreement as list is a %s agreement" % agreement_type)

	doc = corpus.raw("nda-0000-0013.txt")
	agreement_type = classifier.classify_data([doc])
	print("The nda agreement as list is a %s agreement" % agreement_type)

	doc = corpus.raw("nda-0000-0021.txt")
	agreement_type = classifier.classify_data([doc])
	print("The nda agreement as list is a %s agreement" % agreement_type)

	doc = corpus.raw("nda-0000-0022.txt")
	agreement_type = classifier.classify_data([doc])
	print("The nda agreement as list is a %s agreement" % agreement_type)

	doc = corpus.raw("nda-0000-0005.txt")
	agreement_type = classifier.classify_data([doc])
	print("The agreement is a %s agreement" % agreement_type)

	doc = corpus.raw("b9debb6714e6c54e0e3ac431af2215c24784d195418988be65568fd92dfb3fd7")
	agreement_type = classifier.classify_data([doc])
	print("The (convertible) agreement is a %s agreement" % agreement_type)

	doc = corpus.raw("03dbdc3e21e39d5c5a30535fde6768796f19f83305370aeafd774dfd82d8cb9e")
	agreement_type = classifier.classify_data([doc])
	print("The (indenture) agreement is a %s agreement" % agreement_type)

	doc = corpus.raw("80af9aebbb2bf04eae5c011717fb091745867e2f2e7e991b74bd6d92c1639990")
	agreement_type = classifier.classify_data([doc])
	print("The (convertible) agreement is a %s agreement" % agreement_type)

	doc = corpus.raw("55b909a381ce30e45586a68cb93e77622e265cc91d923c36ac31faf8ff37d534")
	agreement_type = classifier.classify_data([doc])
	print("The (revolving credit) agreement is a %s agreement" % agreement_type)

	doc = corpus.raw("nnn-0000-0001.txt")
	agreement_type = classifier.classify_data([doc])
	print("The nnn agreement is a %s agreement" % agreement_type)

	doc = corpus.raw("nnn-0000-0002.txt")
	agreement_type = classifier.classify_data([doc])
	print("The nnn agreement is a %s agreement" % agreement_type)

	doc = corpus.raw("nnn-0000-0003.txt")
	agreement_type = classifier.classify_data([doc])
	print("The nnn agreement is a %s agreement" % agreement_type)

	doc = corpus.raw("nnn-0000-0004.txt")
	agreement_type = classifier.classify_data([doc])
	print("The nnn agreement is a %s agreement" % agreement_type)

	doc = corpus.raw("nnn-0000-0005.txt")
	agreement_type = classifier.classify_data([doc])
	print("The nnn agreement is a %s agreement" % agreement_type)

	doc = corpus.raw("nnn-0000-0006.txt")
	agreement_type = classifier.classify_data([doc])
	print("The nnn agreement is a %s agreement" % agreement_type)

	doc = corpus.raw("nnn-0000-0014.txt")
	agreement_type = classifier.classify_data([doc])
	print("The nnn agreement is a %s agreement" % agreement_type)


def testing_nb():
	print("Loading the datastores...")
	from helper import WiserDatabase
	datastore = WiserDatabase()
	print("Build a corpus of documents...")
	corpus = build_corpus(binary=False, binary_param='nondisclosure')
	print("Some categories were identified...")
	print(corpus.categories())
	print("Loading the agreement classifier...")
	classifier = get_agreement_classifier_v3(corpus)
	print("Application ready to load.")
	"""
	a doc should be a list of strings : [ "string1", "string2", "string3" .. ]
	"""
	doc = corpus.words("nda-0000-0008.txt")
	agreement_type = classifier.classify_data(doc)
	print("The nda agreement is a %s agreement" % agreement_type)

	doc = corpus.raw("nda-0000-0008.txt")
	agreement_type = classifier.classify_data([doc])
	print("The nda agreement as list is a %s agreement" % agreement_type)	

def testing_both():
	print("Loading the datastores...")
	from helper import WiserDatabase
	datastore = WiserDatabase()
	print("Build a corpus of documents...")
	corpus = build_corpus(binary=False, binary_param='nondisclosure')
	print("Some categories were identified...")
	print(corpus.categories())
	print("Loading the agreement classifier...")
	classifier1 = get_agreement_classifier_v1(corpus)
	classifier3 = get_agreement_classifier_v3(corpus)

	doc = corpus.words("nda-0000-0008.txt")
	agreement_type = classifier1.classify_data(doc)
	print("The nda agreement is a %s agreement" % agreement_type)

	doc = corpus.words("nda-0000-0008.txt")
	agreement_type = classifier3.classify_data(doc)
	print("The nda agreement is a %s agreement" % agreement_type)


def main():
	pass

"""
"""
if __name__ == "__main__":
    pass