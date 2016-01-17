#!/usr/bin/python
from sklearn.feature_extraction import DictVectorizer
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.tokenize import BlanklineTokenizer
from sklearn import svm

import os
import nltk
from nltk.tokenize import word_tokenize
from trainer import Trainer

BASE_PATH = "./"
DATA_PATH = os.path.join(BASE_PATH, "data/")
FEATURE_PATH = os.path.join(BASE_PATH, "feature/")
CURATED_PATH = os.path.join(BASE_PATH, "parsed/")

class Feature(object):

	def __init__(self):
		import ner
		import config
		ner_settings = config.load_ner()
		self.tagger = ner.SocketNER(host=ner_settings['hostname'], port=int(ner_settings['port']))

	def classify(content):
		if content and len(content) > 0:
			if isinstance(content[0], str):
				return self.text_identify(content)
			else:
				return self.dict_identify(content)

	
	def load_dict(self):
		""" 
        "document_class": "nondisclosure", 
        "feature_class": "features/feature_preamble", 
        "filename": "nda-0000-0001.txt", 
        "first_guess": "train/train_preamble", 
        "next_guess": "train/train_recital", 
        "paragraph_index": 0, 
        "prev_guess": null, 
		"""
		from helper import WiserDatabase
		datastore = WiserDatabase()
		feature_classes = datastore.fetch_feature_class_types()
		print(feature_classes)
		feature_classes.remove("")
		#feature_classes.remove("features/feature_definitions")
		dict_values = []
		key_values = []
		for feature_class in feature_classes:
			genome = datastore.fetch_curated(feature_class)
			for r in genome: 
				key_values.append(r['feature_class'])
				stats = self.calc_stat(r['text'])
				del r['text']
				del r['feature_class']
				del r['document_class']
				if 'filename' in r.keys():
					del r['filename']

				if not r['first_guess']:
					del r['first_guess']

				if not r['next_guess']:
					del r['next_guess']
				
				if not r['prev_guess']: 
					del r['prev_guess']
	
				stats.update(r)
				dict_values.append(stats)
		
		#import json
		#print(json.dumps(dict_values[0:15], indent=3))
		# dict_values = [{...}, {...}, {...}]
		# key_values = ["features/feature_normal_text", ... ]
		#print(dict_values)
		self.vectorizer = DictVectorizer(sparse=True)
		train_vec = self.vectorizer.fit_transform(dict_values)
		self.cll = svm.LinearSVC(class_weight='auto')
		self.cll.fit(train_vec, key_values)
		print("ready to fit features using curated data")

	def load_trainers(self):
		feature_files = {
			"definitions" : "features/feature_definitions", 
			"signature_line" : "features/feature_signature_line", 
			"title_paragraph" : "features/feature_title_paragraph",
			"normal_text" : "features/feature_normal_text",
			"preamble" : "features/feature_preamble",
			"recital" : "features/feature_recital",
		}

		self.feature_corpus = PlaintextCorpusReader(BASE_PATH, feature_files.values())
		self.vectorizer = DictVectorizer(sparse=True)
		fileids = self.feature_corpus.fileids()

		sents_stats = []
		for fileid in fileids:
			alltext = self.feature_corpus.raw(fileid)
			doc = self.blank_tokenize(alltext)
			stats = self.calc_stats(doc)
			sents_stats += zip([fileid] * len(doc), stats)

		key_values = [d[0] for d in sents_stats]
		dict_values = [d[1] for d in sents_stats]
		train_vec = self.vectorizer.fit_transform(dict_values)

		self.cll = svm.LinearSVC(class_weight='auto')
		self.cll.fit(train_vec, key_values)
		print("ready to fit features using file data")

	def tokenize(self, content):
		tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
		return tokenizer.tokenize(content)

	def blank_tokenize(self, content):
		b = BlanklineTokenizer()
		return b.tokenize(content)

	def calc_stat(self, paragraph):
		nerz = self.tagger.get_entities(paragraph)
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
		return stats

	def calc_stats(self, sent_toks):
		""" calculate statistics about paragraphs.
		:sent_toks: list of strings 
		Returns a list of dicts.
		""" 
		para_meta = []
		for paragraph in sent_toks:
			stats = self.calc_stat(paragraph)
			para_meta.append(stats)
		return para_meta

	def build_dict(self, tupleized):
		import nltk
		pidx = 0
		provision_count = len(tupleized)
		allinfo = []
		for (_block, _type) in tupleized:
			property_dict = dict()
			#property_dict['filename'] = filename
			property_dict['document_class'] = "nondisclosure"
			property_dict['first_guess'] = _type
			property_dict['paragraph_index'] = pidx
			property_dict['relative_paragraph_index'] = round(float(100) * (float(pidx) / provision_count), 2)
			#property_dict['feature_class'] = provision_features[pidx][1]
			property_dict['character_count'] = len(_block)
			#property_dict['text'] = _block

			words = nltk.tokenize.word_tokenize(_block)
			property_dict['word_count'] = len(words)

			nerz = self.tagger.get_entities(_block)
			types = ["DATE", "ORGANIZATION", "PERSON", "LOCATION","MONEY", "PERCENT", "TIME"]
			missing = set(types) - set(nerz.keys())
			for k in missing:
				property_dict["contains" + k] = False

			for k in nerz.keys():
				if nerz[k]:
					property_dict["contains" + k] = True

			if pidx > 0 and pidx < provision_count-1:
				property_dict['next_guess'] = tupleized[pidx+1][1]
				property_dict['prev_guess'] = tupleized[pidx-1][1]
			else:
				if pidx == 0:
					property_dict['next_guess'] = tupleized[pidx+1][1]
					property_dict['prev_guess'] = None
				elif pidx == provision_count-1:
					property_dict['next_guess'] = None
					property_dict['prev_guess'] = tupleized[pidx-1][1]

			stats = self.calc_stat(_block)
			property_dict.update(stats)

			allinfo.append(property_dict)
			pidx += 1
		return allinfo	

	def dict_identify(self, tupleized):
		doc_dict = self.build_dict(tupleized)
		test_data = self.vectorizer.transform(doc_dict)
		results = self.cll.predict(test_data)
		content = [_block for (_block, _type) in tupleized]
		tupleized = list(zip(content, list(results)))
		return tupleized

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
	f.load_trainers()

	sents = []
	sents.append("MUTUAL NON-DISCLOSURE")
	sents.append("PROMISSORY NOTE OBLIGATION")
	sents.append("In addition to the above, Confidential Information shall also include, and the parties shall have a duty to protect, other confidential and/or sensitive information which is (a) disclosed as such in writing and marked as confidential (or with other similar designation) at the time of disclosure; and/or (b) disclosed by in any other manner and identified as confidential at the time of disclosure and is also summarized and designated as confidential in a written memorandum delivered within thirty (30) days of the disclosure.")
	sents.append("Mutual Non-Disclosure Agreement\n\n")
	sents.append("MUTUAL NON-DISCLOSURE")
	sents.append("The party disclosing the Confidential Information shall be referred to as\n\"Disclosing Party\" in the Agreement and the party receiving the Confidential Information shall be referred to as\n\"Receiving Party\" in the Agreement.")
	sents.append("Confidential Information: Confidential information means any information disclosed to by one party to the other,\neither directly or indirectly in writing, orally or by inspection of tangible or intangible objects, including without\nlimitation documents, business plans, source code, software, documentation, financial analysis, marketing plans,\ncustomer names, customer list, customer data.")
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

def testing2():
	f = Feature()
	f.load_dict()

	sents = []
	sents.append("MUTUAL NON-DISCLOSURE")
	sents.append("PROMISSORY NOTE OBLIGATION")
	sents.append("In addition to the above, Confidential Information shall also include, and the parties shall have a duty to protect, other confidential and/or sensitive information which is (a) disclosed as such in writing and marked as confidential (or with other similar designation) at the time of disclosure; and/or (b) disclosed by in any other manner and identified as confidential at the time of disclosure and is also summarized and designated as confidential in a written memorandum delivered within thirty (30) days of the disclosure.")
	sents.append("Mutual Non-Disclosure Agreement\n\n")
	sents.append("MUTUAL NON-DISCLOSURE")
	sents.append("The party disclosing the Confidential Information shall be referred to as\n\"Disclosing Party\" in the Agreement and the party receiving the Confidential Information shall be referred to as\n\"Receiving Party\" in the Agreement.")
	sents.append("Confidential Information: Confidential information means any information disclosed to by one party to the other,\neither directly or indirectly in writing, orally or by inspection of tangible or intangible objects, including without\nlimitation documents, business plans, source code, software, documentation, financial analysis, marketing plans,\ncustomer names, customer list, customer data.")
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

