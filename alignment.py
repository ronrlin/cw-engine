#!/usr/bin/python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.corpus.reader.plaintext import CategorizedPlaintextCorpusReader
from sklearn import svm

import os
import nltk
from nltk.tokenize import word_tokenize
from structure import AgreementSchema
from structure import load_training_data
from structure import get_provision_name_from_file
from trainer import Trainer
from feature import Feature

BASE_PATH = "./"
DATA_PATH = os.path.join(BASE_PATH, "data/")

COUNT_VECT = 1
TFIDF_VECT = 2

"""
Alignment describes the process by which agreements of the same kind of are compared
to one another in a way that "aligns" the provisions within them.  Aligning agreements
is critical to being able to calculate relative frequency of certain provisions.  

"""
class Alignment(object):

    def __init__(self, schema=None, vectorizer=COUNT_VECT, stop_words=None, all=False):
        """Create an Alignment object. 

        :param schema: specify the AgremeentSchema to use for alignment.
        :param vectorizer: specify the type of vectorization method.
        :param stop_words: specify stop words to drop from texts.
        :param all: boolean that denotes whether to load all possible 
            provision trainers
        """
        self.schema = schema
        self.concept_dict = None
        provisions = None

        if (not all):
            print("Load %s agreement training provisions" % schema.get_agreement_type())
            provisions = schema.get_provisions()
        else:
            # to load all files
            provisions = load_training_data().items()

        # provisions is a tuple (provision_name, provision_path)
        training_file_names = [p[1] for p in provisions]
        provision_names = [p[0] for p in provisions]

        import time
        start_time = time.time()
        self.training_corpus = PlaintextCorpusReader(BASE_PATH, training_file_names)
        end_time = time.time()
        print("Corpus is loading %s files" % str(len(self.training_corpus.fileids())))
        print("Time to load Plaintext training corpus is %s seconds" % (end_time - start_time))

        from helper import WiserDatabase
        self.datastore = WiserDatabase()
        records = self.datastore.fetch_by_category(schema.get_agreement_type())
        fileids = [r['filename'] for r in records]
        self.agreement_corpus = PlaintextCorpusReader(DATA_PATH, fileids)
        print("Agreement Corpus of type %s has been loaded." % schema.get_agreement_type())

        """
        max_df is parameter to CountVectorizer
        min_df is param to CV

        """
        if (vectorizer == COUNT_VECT): 
            self.vectorizer = CountVectorizer(input='content', stop_words='english', ngram_range=(1,3))
            self.vectorizer2 = CountVectorizer(input='content', stop_words='english', ngram_range=(1,3))
        elif (vectorizer == TFIDF_VECT):
            self.vectorizer = TfidfVectorizer(input='content', stop_words='english', ngram_range=(1,3))
            self.vectorizer2 = TfidfVectorizer(input='content', stop_words='english', ngram_range=(1,3))

        # Try using BlanklineTokenizer
        start_time = time.time()
        from nltk.tokenize import BlanklineTokenizer
        tokenizer = BlanklineTokenizer()
        target2 = []
        train_sents2 = []
        for fileid in self.training_corpus.fileids():
            doc = self.training_corpus.raw(fileid)
            doc = tokenizer.tokenize(doc)
            target2 += [fileid] * len(doc)
            train_sents2 += doc
        end_time = time.time()
        print("Time to use blankline tokenizer on sentences of training texts is %s seconds" % (end_time - start_time))        

        start_time = time.time()
        train_vec2 = self.vectorizer2.fit_transform(train_sents2)
        end_time = time.time()
        print("Time to fit/transform vector is %s seconds" % (end_time - start_time))

        # TODO: Some of the sents() are really small.  
        start_time = time.time()
        train_sents = list(' '.join(s) for s in self.training_corpus.sents())
        end_time = time.time()
        print("Time to load join on sentences of training texts is %s seconds" % (end_time - start_time))

        start_time = time.time()
        train_vec = self.vectorizer.fit_transform(train_sents)
        end_time = time.time()
        print("Time to fit/transform vector is %s seconds" % (end_time - start_time))

        start_time = time.time()
        target = list()
        for tfile in self.training_corpus.fileids():
            for tpara in self.training_corpus.sents(tfile):  
                # TODO: We should really assemble the train_vec here, and maybe combine 
                # short sentences or something like that!! 
                res = [fi for (name, fi) in provisions if (fi==tfile)]
                target.append(res[0])
        end_time = time.time()
        print("Time to assemble a target vector is %s seconds" % (end_time - start_time))

        start_time = time.time()
        self.cll = svm.LinearSVC(class_weight='auto')
        self.cll.fit(train_vec, target)

        self.cll2 = svm.LinearSVC(class_weight='auto')
        self.cll2.fit(train_vec2, target2)
        end_time = time.time()
        print("Time to build classifier and fit is %s seconds" % (end_time - start_time))
        print("\nReady for alignment!")

    def simplify(self, paras):
        """ Function takes a list of strings, joins 'short' strings with others. """
        runAgain = True
        while runAgain is True:
            runAgain = False
            for idx, paragraph in enumerate(paras):
                characters = len(paragraph)
                words = nltk.tokenize.word_tokenize(paragraph)
                if (len(words) <= 10 and idx < len(paras) - 1): #short sentences
                    nextstr = paras.pop(idx+1)
                    paras.insert(idx, paragraph + " " + nextstr)
                    discarded = paras.pop(idx + 1)
                    runAgain = True
        return paras

    def aligncore(self, content, version=1):
        tupleized = []
        if version == 1:
            test_vec = self.vectorizer.transform(content)
            results = self.cll.predict(test_vec)
            tupleized = list(zip(content, list(results)))
        elif version == 2:
            test_vec = self.vectorizer2.transform(content)
            results = self.cll2.predict(test_vec)
            tupleized = list(zip(content, list(results)))
        return tupleized

    def align(self, content, version=1):
        """ The smartest part.
        :param content: a list of strings 
        """
        print("using version %s (2 = Blankline tokenizer)" % str(version))
        tupleized = self.aligncore(content=content, version=version)

        feature = Feature()
        self.provision_features = feature.text_identify(content)
        #print([f[1] for f in self.provision_features])

        concepts = self.schema.get_concepts()
        concept_keys = [a[0] for a in concepts]
        provisions = self.schema.get_provisions()

        #unique_provs = set(results)
        self.concept_dict = dict.fromkeys(concept_keys, [])
        for c in concepts: 
            ctr = 0
            index = 0
            for (_text, _type) in tupleized:
                if (c[0] == _type or c[0] in _type):
                    dict_val = {}
                    if ("concept/train_" in c[1]):
                        # Some concepts require loading a trainer
                        fileids = c[1].replace(" ", "").split(",")
                        concept_trainer = Trainer(fileids=fileids)
                        _class = concept_trainer.classify_text(_text)
                        _class = _class.replace("concept/train_", "")
                        dict_val = { 'class' : _class.replace("_", "-"), 'index' : index, 'start' : 0, 'ctr' : ctr, 'len' : _text.find(".") - 1 }
                        provkey = _type.replace("train/train_", "")
                        self.concept_dict[provkey].append(dict_val)
                        ctr = ctr + 1 # this tells you how many times you've found this type of provision
                    else: 
                        # Some concepts can be obtained otherwise
                        for con in c[1:]:
                            dict_val = self._markup_concepts(concept_name=con, text=_text, index=index, counter=ctr)
                            provkey = _type.replace("train/train_", "")
                            self.concept_dict[provkey].append(dict_val)
                            ctr = ctr + 1 # this tells you how many times you've found this type of provision
                index = index + 1
        return tupleized

    def sanity_check(self, tupleized):
        """ Suppresses bad classification """ 
        #provision_features = [("some text", 'features/feature_title_paragraph'),("some text", 'features/feature_title_paragraph', "some text")]
        provision_features = self.provision_features
        new_tupleized = []
        for i, (provision, _type) in enumerate(tupleized):
            feature_type = provision_features[i][1]
            if ("title_paragraph" in feature_type):
                new_tupleized.append((provision, ""))
            elif ("signature_line" in feature_type):
                new_tupleized.append((provision, ""))
            elif ("definitions" in feature_type):
                new_tupleized.append((provision, _type))
            elif ("normal_text" in feature_type):
                new_tupleized.append((provision, _type))
            else:
                new_tupleized.append((provision, _type))
        return new_tupleized

    def _markup_concepts(self, concept_name, text, index, counter=0):
        """ This is trivial.  Put a span around exact matches 
        of the 'concept' in the text.  The 'concept' is just
        the concept_name with underscores removed.
        """
        concept_str = concept_name.replace("_", " ")
        # the index marks the index in tupleized where the concept is found.
        _index = index
        # the counter keeps track of the incidence of this concept.
        _ctr = counter
        _class = concept_name.replace("_", "-")
        _start = text.find(concept_str)
        _len = len(concept_str)
        values = { 'class' : _class, 'index' : _index, 'start' : _start, 'ctr' : _ctr, 'len' : _len }
        return values

    def tokenize(self, content):
        """ 
        :param content: is a string that is to be tokenized
        return list of strings
        """
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        return tokenizer.tokenize(content)

    def get_markup(self, tupleized):
        """ returns content with markup to identify provisions within agreement """
        _markup_list = []
        concept_provs = self.concept_dict.keys()
        inc = dict((y,0) for (x, y) in tupleized)
        for (_block, _type) in tupleized:
            text = _block
            if not _type:
                text = "<p><div>" + text + "</div></p>"
            else: 
                # first add concepts
                if (get_provision_name_from_file(_type) in concept_provs):
                    values = self.concept_dict[get_provision_name_from_file(_type)]
                    if values:
                        tc = values.pop(0)
                        text = text[:tc['start']] + "<span id='concept-" + tc['class'] + "-" + str(tc['ctr']) + "' class='concept " + tc['class'] + "'>" + text[tc['start']:tc['start']+tc['len']] + "</span>" + text[tc['start']+tc['len']:]

                # then wrap in provision markup
                text = "<div id='provision-" + get_provision_name_from_file(_type, True) + "-" + str(inc[_type]) + "' class='provision " + get_provision_name_from_file(_type, True) + "'>" + text + "</div>"
                text = "<p>" + text + "</p>"

            _markup_list.append(text)
            inc[_type] = inc[_type] + 1
        return " ".join(_markup_list)

    def get_tags(self, document):
        tags = self.schema.get_tags()
        tupled = []
        output = []
        for tag in tags:
            # tag is a tuple
            tag_name = tag[0]
            tag_values = tag[1].split(",") # list of all possible values
            for val in tag_values:
                val = val.strip(" ")
                #fileids = self.datastore.fetch_by_classified_tag(tag_name, val)
                fileids = self.datastore.fetch_by_tag({tag_name : val})
                thistuple = (zip(fileids, [val] * len(fileids)))
                #need to append elements of thistuple to tupled
                for t in thistuple:
                    tupled.append(t)

            print("%s files will be loaded into tag corpus." % str(len(tupled)))   
            result = {}
            if (len(tupled) > 1):
                mapped = dict(tupled)
                tagged_corpus = CategorizedPlaintextCorpusReader(DATA_PATH, fileids=mapped.keys(), cat_map=mapped)
                vectorizer = TfidfVectorizer(input='content', stop_words=None, ngram_range=(1,2))
                from classifier import AgreementVectorClassifier 
                classifier = AgreementVectorClassifier(vectorizer, tagged_corpus)
                classifier.fit()
                result['type'] = tag_name
                result['category'] = classifier.classify_data(document)
                result['reference-info'] = '' #some id into a reference db
                result['text'] = '' #TODO: this is how the tags get displayed
            else:
                print("problem in get_tag!")
                result['type'] = ""
                result['category'] = ""
                result['reference-info'] = '' #some id into a reference db
                result['text'] = '' #TODO: this is how the tags get displayed
            output.append(result)
        return output

    def get_concept_detail(self):
        concept_detail = {}
        allconcepts = self.schema.get_concepts()
        for c in allconcepts:
            c = c[1]
            c = c.replace(" ", "")
            c = c.split(",")
            for concepts in c:
                concept_class = concepts.replace("concept/train_", "")
                concept_class = concept_class.replace("_", "-")
                concept_detail[concept_class] = { "description" : "this will say something about %s" % concept_class, "title" : concept_class}
        return concept_detail

    def get_detail(self, tupleized):
        # Collect contract_group statistics from datastore
        contract_group = self.datastore.get_contract_group(self.schema.get_agreement_type()) 

        from statistics import AgreementStatistics
        astats = AgreementStatistics(tupleized)
        aparams = astats.calculate_stats()
        # doc is the text of the agreement, formed by joining all the text blocks in the tuple
        doc = [e[0] for e in tupleized]
        doc = " ".join(doc)
        
        document = dict()
        document['mainDoc'] = {
            '_body' : self.get_markup(tupleized),
            'agreement_type' : self.schema.get_agreement_type(), # get this from contract_group_info
            'text-compare-count' : len(self.agreement_corpus.fileids()), # get this from contract_group_info
            # doc-similarity is this doc compared to the group
            'doc-similarity-score' : astats.calculate_similarity(doc, self.agreement_corpus), # contract_group['doc-similarity-score'] get this from contract_group_info 
            'doc-complexity-score' : aparams['doc_gulpease'],
            'group-similarity-score' : round(contract_group['group-similarity-score'], 1), # get this from contract_group_info
            'group-complexity-score' : round(contract_group['group-complexity-score'], 1), # get this from contract_group_info
            'tags' : self.get_tags(doc),
        }

        provisions = {}

        print("scroll through tupleized")
        for (_block, _type) in tupleized:
            # Collect provision_group statistics from datastore
            provision_mach_name = get_provision_name_from_file(_type, dashed=False)
            provision_name = get_provision_name_from_file(_type, dashed=True)
            provision_group_info = self.datastore.get_provision_group(provision_mach_name)
            if provision_group_info is not None:
                provisions[provision_name] = {
                    'provision-readable' : provision_name,
                    'consensus-percentage' : astats.get_consensus(self.agreement_corpus, _type),
                    "prov-similarity-score" : astats.calculate_similarity(_block, self.training_corpus), # needs works!
                    "prov-similarity-avg" : round(provision_group_info['prov-similarity-avg'], 1), # get this from provision_group_info
                    "prov-complexity-score" : astats.calculate_complexity(_block), # computed on the fly
                    "prov-complexity-avg" : round(provision_group_info['prov-complexity-avg'], 1), # get this from provision_group_info
                    #"provision-tag" : "some-label", # computed on the fly
                }
            else: 
                # TODO: log an error here and 
                # raise an error about not having data for this
                provisions[provision_name] = {}

        document['provisions'] = provisions
        document['concepts'] = self.get_concept_detail()
        return document

def testing(filename="nda-0000-0014.txt"):
    """ test that the class is working """
    schema = AgreementSchema()
    print("loading the nondisclosure schema...")
    schema.load_schema("nondisclosure.ini")
    provisions = schema.get_provisions()
    print("we're looking for...")
    print(provisions)
    a = Alignment(schema=schema)
    #filename = "nda-0000-0014.txt"
    from classifier import build_corpus
    corpus = build_corpus()
    doc = corpus.raw(filename)
    print("tokenize raw text into sentences.")
    toks = a.tokenize(doc)
    print("combine really short sentences.")
    toks = a.simplify(toks)
    print("alignment...")
    result = a.align(toks)
    print("check on the markup")
    markup = a.get_markup(result)
    print("returns json")
    import json
    print(json.dumps(a.get_detail(result)))
    print("what features we found")
    print([f[1] for f in a.provision_features])

    """ example of some simple chunking """ 
    #for sent in testset: 
    #    for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
    #        if hasattr(chunk, 'label'):
    #            print(chunk.label(), ' '.join(c[0] for c in chunk.leaves()))

def testr():
    filename = "nda-0000-0034.txt"
    print("obtain a corpus...")
    from classifier import build_corpus
    corpus = build_corpus()
    from structure import AgreementSchema
    schema = AgreementSchema()
    schema.load_schema('nondisclosure')
    #from Alignment import alignment
    aligner = Alignment(schema=schema, vectorizer=2, all=False)
    doc = corpus.raw(filename)
    paras = aligner.tokenize(doc)
    aligned_provisions = aligner.align(paras) # aligned_provisions is a list of tuples
    print("aligned provisions")
    print(aligned_provisions)
    print("what features we found")
    print([f[1] for f in aligner.provision_features])
    res = aligner.sanity_check(aligned_provisions)
    print("sanity check")
    print(res)
    #results = aligner.get_markup(aligned_provisions)
    #print(results)
    #print("\naligned provisions\n")
    #print(aligned_provisions)
    #print("\n")
    #tupleized = aligner.continguous_normalize(aligned_provisions)
    #print(tupleized)
    #return tupleized

def comp():
    print("obtain a corpus...")
    from structure import AgreementSchema
    from classifier import build_corpus
    schema = AgreementSchema()
    schema.load_schema('nondisclosure')
    corpus = build_corpus()
    doc = corpus.raw("nda-0000-0014.txt")

    aligner1 = Alignment(schema=schema, vectorizer=TFIDF_VECT, all=False)
    aligner2 = Alignment(schema=schema, vectorizer=TFIDF_VECT, all=True)

    paras = aligner1.tokenize(doc)
    aligned_provisions1 = aligner1.align(paras) # aligned_provisions is a list of tuples
    print(aligned_provisions1)
    paras = aligner2.tokenize(doc)
    aligned_provisions2 = aligner2.align(paras) # aligned_provisions is a list of tuples

    return(aligner1, aligner2)

def build_provision_tests():
    provisions = []
    ##
    ##
    provision_text = """2. Confidential Information: Confidential information means any information disclosed to by one party to the other,
either directly or indirectly in writing, orally or by inspection of tangible or intangible objects, including without
limitation documents, business plans, source code, software, documentation, financial analysis, marketing plans,
customer names, customer list, customer data. Confidential Information may also include information disclosed to a
party by third parties at the direction of a Disclosing Party. Confidential Information shall not, however, include any
information which the Receiving party can establish (i) was publicly known and made generally available in the public
domain prior to the time of disclosure; (ii) becomes publicly known and made generally available after disclosure
through no action or inaction of Receiving Party; or (iii) is in the possession of Receiving Party, without confidentiality
restrictions, at the time of disclosure by the Disclosing Party as shown by Receiving Party's files and records
immediately prior to the time of disclosure. The party disclosing the Confidential Information shall be referred to as
"Disclosing Party" in the Agreement and the party receiving the Confidential Information shall be referred to as
"Receiving Party" in the Agreement.    
    """
    provisions.append(provision_text)
    ##
    ##
    provision_text = """
    2. Each party acknowledges and agrees that the Confidential Information of the other party is a valuable asset of such party and has competitive value.   
    """
    provisions.append(provision_text)    
    ##
    ##
    provision_text = """4. The term "Confidential Information" shall not include information which (a) is or becomes generally available to the public
other than as a result of a disclosure by the Recipient or its Representatives, (b) is or becomes available to the Recipient
from a source other than the Disclosing Party or its Representatives, provided that such source obtained such information
lawfully and is not, and was not, bound by a confidentiality agreement with, or obligation to, the Disclosing Party or any of its
affiliates or Representatives. """
    provisions.append(provision_text)
    ##
    ##
    provision_text = """
        13. This Agreement shall survive the termination of any negotiations or discussions between the
        parties hereto for a period of two years and may not be modified or terminated, in whole or in part, and no release
        hereunder shall be effective except by means of a written instrument executed by the parties hereto.
    """
    ##
    ##
    provision_text = """1. Definition of Confidential Information. For purposes of this Agreement, "Confidential Information" shall include all information or material that has or could have commercial value or other utility in the business in which Disclosing Party is engaged. If Confidential Information is in written form, the Disclosing Party shall label or stamp the materials with the word "Confidential" or some similar warning. If Confidential Information is transmitted orally, the Disclosing Party shall promptly provide a writing indicating that such oral communication constituted Confidential Information.
    """
    provisions.append(provision_text)
    ##
    ##
    provision_text = """5.    In the event that the Recipient shall breach this Agreement, or in the event that a breach appears to be imminent, the Disclosing Party shall be entitled to all legal and equitable remedies afforded it by law, and in addition may recover all reasonable costs and attorneys' fees incurred in seeking such remedies.  If the Confidential Information is sought by any third party, including by way of subpoena or other court process, the Recipient shall inform the Disclosing Party of the request in sufficient time to permit the Disclosing Party to object to and, if necessary, seek court intervention to prevent the disclosure.
    """    
    provisions.append(provision_text)  
    ##
    ##
    provision_text = """Basic Nondisclosure Agreement"""
    provisions.append(provision_text)
    ##
    ##
    provision_text = """
This Nondisclosure Agreement (the "Agreement") is entered into by and between _______________ with its principal offices at _______________ ("Disclosing Party") and _______________, located at _______________ ("Receiving Party") for the purpose of preventing the unauthorized disclosure of Confidential Information as defined below. The parties agree to enter into a confidential relationship with respect to the disclosure of certain proprietary and confidential information ("Confidential Information")."""    
    provisions.append(provision_text)  

    return provisions  

def newtest(version=1, sanity=False):
    """ test that the class is working """
    schema = AgreementSchema()
    print("loading the nondisclosure schema...")
    schema.load_schema("nondisclosure.ini")
    provisions = schema.get_provisions()
    print("we're looking for...")
    print(provisions)
    a = Alignment(schema=schema)
    provisions = build_provision_tests()

    result = None
    if version == 1:
        print("using version 1")
        result = a.align(provisions, version=1)
    elif version == 2:
        print("using version 2")
        result = a.align(provisions, version=2)

    if sanity:
        print(">> sane results <<")
        result = a.sanity_check(result)
    else:
        print(">> normal results <<")

    for i, r in enumerate(result):
        print("=====")
        print(r[0])
        print(">>" + r[1] + " | " + a.provision_features[i][1])

"""
>>> from sklearn.feature_extraction import DictVectorizer
>>> v = DictVectorizer(sparse=False)
>>> D = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]
>>> X = v.fit_transform(D)
>>> X
array([[ 2.,  0.,  1.],
       [ 0.,  1.,  3.]])
>>> v.inverse_transform(X) ==         [{'bar': 2.0, 'foo': 1.0}, {'baz': 1.0, 'foo': 3.0}]
True
>>> v.transform({'foo': 4, 'unseen_feature': 3})
"""

"""
Bypass main
"""
if __name__ == "__main__":
    pass