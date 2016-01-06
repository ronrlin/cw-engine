#!/usr/bin/python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.corpus.reader.plaintext import CategorizedPlaintextCorpusReader
from nltk.tokenize import BlanklineTokenizer
from sklearn import svm

import os
import nltk
from nltk.tokenize import word_tokenize
from structure import AgreementSchema
from structure import load_training_data
from structure import get_provision_name_from_file
from trainer import Trainer

BASE_PATH = "./"
DATA_PATH = os.path.join(BASE_PATH, "data/")
FEATURE_PATH = os.path.join(BASE_PATH, "feature/")

class Feature(object):

	def __init__(self):
		print("init feature loading...")
		# TODO: make this dynamic
		feature_files = {
			"definitions" : "features/feature_definitions", 
			"signature_line" : "features/feature_signature_line", 
			"title_paragraph" : "features/feature_title_paragraph",
			"normal_text" : "features/feature_normal_text",
			"preamble" : "features/feature_preamble",
			"recital" : "features/feature_recital",
		}

		import time
		start_time = time.time()
		self.feature_corpus = PlaintextCorpusReader(BASE_PATH, feature_files.values())
		end_time = time.time()
		print("Corpus is loading %s files" % str(len(self.feature_corpus.fileids())))
		print("Time to load Plaintext training corpus is %s seconds" % (end_time - start_time))

		self.vectorizer = DictVectorizer(sparse=True)

		start_time = time.time()
		fileids = self.feature_corpus.fileids()

		tagged_sents = []
		sents_stats = []
		docs = []
		for fileid in fileids:
			alltext = self.feature_corpus.raw(fileid)
			doc = self.blank_tokenize(alltext)
			stats = self.calc_stats(doc)
			tagged_sents += zip([fileid] * len(doc), doc)
			sents_stats += zip([fileid] * len(doc), stats)

		end_time = time.time()
		print("Time to load join on sentences of feature texts is %s seconds" % (end_time - start_time))

		# tuple
		# tagged_sents = [('filename', [u'CONFIDENTIALITY', u'AGREEMENT']), ... ]
		# sents_stats = [('filename', {'':'', '':'', ... }),]
		# sents_stats[fileid] = {}

		key_values = [d[0] for d in sents_stats]		
		dict_values = [d[1] for d in sents_stats]
		start_time = time.time()
		train_vec = self.vectorizer.fit_transform(dict_values)
		end_time = time.time()
		print("Time to fit/transform vector is %s seconds" % (end_time - start_time))

		start_time = time.time()
		self.cll = svm.LinearSVC(class_weight='auto')
		self.cll.fit(train_vec, key_values)
		end_time = time.time()
		print("Time to build classifier and fit is %s seconds" % (end_time - start_time))
		print("\nReady for alignment!")

	def tokenize(self, content):
		tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
		return tokenizer.tokenize(content)

	def blank_tokenize(self, content):
		b = BlanklineTokenizer()
		return b.tokenize(content)

	# TODO: you can eliminate position_flag
	# pass the pos_arr with the values that need to be added for position
	def calc_stats(self, sent_toks, position_flag=False):
		""" calculate statistics about paragraphs.
		:sent_toks: list of strings 

		might want to consider a position_flag=False
		to ignore position for training files
		""" 
		import ner
		import config
		ner_settings = config.load_ner()
		tagger = ner.SocketNER(host=ner_settings['hostname'], port=int(ner_settings['port']))

		para_meta = []
		index = 0
		for paragraph in sent_toks:
			nerz = tagger.get_entities(paragraph)
			stats = {}
			stats['characters'] = len(paragraph)
			stats['returns'] = paragraph.count("\n") #returns with few characters suggests titles or tables
			stats['period'] = paragraph.count(".") # sentences
			stats['asterisks'] = paragraph.count("*") # sentences
			stats['ellipse'] = paragraph.count("...")
			stats['dashes'] = paragraph.count("-")
			stats['commas'] = paragraph.count(",") #commas are a sign of complex sentences
			stats['semis'] = paragraph.count(";") #commas are a sign of complex sentences
			stats['underscores'] = paragraph.count("_")
			stats['colon'] = paragraph.count(":")
			stats['new_line'] = paragraph.count("\n")			
			stats['hasAgreement'] = "agreement" in paragraph.lower()
			stats['hasAgreementCaps'] = "AGREEMENT" in paragraph
			stats['hasBetween'] = "between" in paragraph.lower()
			stats['hasBetweenCaps'] = "BETWEEN" in paragraph
			stats['hasSignature'] = "signature" in paragraph.lower()
			stats['hasBy'] = "by" in paragraph.lower()
			stats['by_count'] = paragraph.count("By")
			stats['hasPrintedName'] = "printed name" in paragraph.lower()
			stats['hasWitnessWhereof'] = paragraph.count("IN WITNESS WHEREOF")
			stats['hasWhereas'] = "whereas" in paragraph.lower()
			stats['hasWhereasCaps'] = "WHEREAS" in paragraph
			stats['hasEffectiveDate'] = "effective date" in paragraph.lower()
			stats['defineEffectiveDate'] = "\"Effective Date\"" in paragraph

			#TODO: you might want to use some NER here.  
			#For example, containsORGANIZATION, containsDATE, containsMONEY
			if "MONEY" in nerz.keys():
				stats['contains' + "MONEY"] = True
			else: 
				stats['contains' + "MONEY"] = False

			if "ORGANIZATION" in nerz.keys():
				stats['contains' + "ORGANIZATION"] = True
			else: 
				stats['contains' + "ORGANIZATION"] = False

			if "LOCATION" in nerz.keys():
				stats['contains' + "LOCATION"] = True
			else: 
				stats['contains' + "LOCATION"] = False

			if "DATE" in nerz.keys():
				stats['contains' + "DATE"] = True
			else: 
				stats['contains' + "DATE"] = False

			#TODO: might want to consider incidence of cardinal numbers
			# while using nltk.ne_chunk( looking for CD )

			if position_flag:
				stats['position_index'] = index
				if index > 0:
					stats['colon_prev'] = para_meta[index-1]['colon']
				else:
					stats['colon_prev'] = 0
			else:
				pass

			stats['bracket_open'] = paragraph.count("[")
			stats['bracket_close'] = paragraph.count("]")
			stats['paren_open'] = paragraph.count("(")
			stats['paren_close'] = paragraph.count(")")
			stats['contains_list'] = paragraph.count("(i)") + paragraph.count("(a)") + paragraph.count("(1)") + paragraph.count("1.") + paragraph.count("A.")
			words = nltk.tokenize.word_tokenize(paragraph)
			uppercase = 0
			word_count = 0
			for word in words:
				if word.isupper():
					uppercase += 1
				if len(word) > 4: ## would be nice to only count certain things as words
					word_count = word_count + 1
			stats['uppercase'] = uppercase
			para_meta.append(stats)
			index = index + 1
		return para_meta

	def text_identify(self, content):
		""" Function determines what kind of text feature this is. 
		ie: signature_line, title, a table, a list, a normal paragraph, etc...
		"""
		dict_values = self.calc_stats(content)
		test_vec = self.vectorizer.transform(dict_values)
		results = self.cll.predict(test_vec)
		tupleized = list(zip(content, list(results)))
		return tupleized


def testing():
	f = Feature()

	sents = []
	sents.append("MUTUAL NON-DISCLOSURE")
	sents.append("PROMISSORY NOTE OBLIGATION")
	sents.append("In addition to the above, Confidential Information shall also include, and the parties shall have a duty to protect, other confidential and/or sensitive information which is (a) disclosed as such in writing and marked as confidential (or with other similar designation) at the time of disclosure; and/or (b) disclosed by in any other manner and identified as confidential at the time of disclosure and is also summarized and designated as confidential in a written memorandum delivered within thirty (30) days of the disclosure.")
	sents.append("Mutual Non-Disclosure Agreement\n\n")
	sents.append("MUTUAL NON-DISCLOSURE")
	sents.append("By: _________________________________	\nBy: _________________________________")
	sents.append(u"MUTUAL NON-DISCLOSURE AGREEMENT\n\nTHIS AGREEMENT is made on\t[Month, day, year]\n\nBETWEEN\n[Party A], ('Party A'); and \n[Party B], ('Party B'), \ncollectively referred to as the 'Parties'.")
	sents.append("""
		COMPANY:
		______________________________________
		 U.S. BANK:
		______________________________________
		By:
		By: 
	""")

	print("the result should be a tuple")
	print(f.text_identify(sents))

