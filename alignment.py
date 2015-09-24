#!/usr/bin/python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
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
            self.vectorizer = CountVectorizer(input='content', stop_words='english', ngram_range=(1,2))
        elif (vectorizer == TFIDF_VECT):
            self.vectorizer = TfidfVectorizer(input='content', stop_words=None, ngram_range=(1,2))

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
        end_time = time.time()
        print("Time to build classifier and fit is %s seconds" % (end_time - start_time))
        print("\nReady for alignment!")

    def align(self, content):
        test_vec = self.vectorizer.transform(content)
        results = self.cll.predict(test_vec)
        tupleized = list(zip(content, list(results)))

        concepts = self.schema.get_concepts()
        concept_keys = [a[0] for a in concepts]
        provisions = self.schema.get_provisions()

        unique_provs = set(results)
        self.concept_dict = dict.fromkeys(concept_keys, [])
        for c in concepts: 
            ctr = 0
            index = 0
            for (_text, _type) in tupleized:
                if (c[0] == _type or c[0] in _type):
                    dict_val = {}
                    if ("train/train_" in c[1]):
                        # Some concepts require loading a trainer
                        fileids = c[1].replace(" ", "").split(",")
                        concept_trainer = Trainer(fileids=fileids)
                        _class = concept_trainer.classify_text(_text)
                        dict_val = { 'class' : _class.replace("_", "-"), 'index' : index, 'start' : 0, 'ctr' : ctr, 'len' : _text.find(".") - 1 }
                        provkey = _type.replace("train/train_", "")
                        self.concept_dict[provkey].append(dict_val)
                        ctr = ctr + 1 # this tells you how many times you've found this type of provision
                    else: 
                        # Some concepts can be obtained otherwise
                        for con in c[1:]:
                            dict_val = self._markup_concepts(con, text, index, ctr)
                            self.concept_dict[_type].append(dict_val)
                            ctr = ctr + 1 # this tells you how many times you've found this type of provision
                index = index + 1
        print(self.concept_dict)
        return tupleized

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
        index = 0
        concept_provs = self.concept_dict.keys()
        inc = dict((y,0) for (x, y) in tupleized)
        for (_block, _type) in tupleized:
            text = _block
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
            index = index + 1
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
                fileids = self.datastore.fetch_by_classified_tag(tag_name, val)
                thistuple = (zip(fileids, [val] * len(fileids)))
                #need to append elements of thistuple to tupled
                for t in thistuple:
                    tupled.append(t)

            print("%s files will be loaded into corpus." % str(len(tupled)))   
            mapped = dict(tupled)
            tagged_corpus = CategorizedPlaintextCorpusReader(DATA_PATH, fileids=mapped.keys(), cat_map=mapped)
            vectorizer = TfidfVectorizer(input='content', stop_words=None, ngram_range=(1,2))
            from classifier import AgreementVectorClassifier 
            classifier = AgreementVectorClassifier(vectorizer, tagged_corpus)
            classifier.fit()
            result = {}
            result['type'] = tag_name
            result['category'] = classifier.classify_data(document)
            result['reference-info'] = '' #some id into a reference db
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
                concept_class = concepts.replace("train/train_", "")
                concept_class = concept_class.replace("_", "-")
                concept_detail[concept_class] = { "description" : "this will say something about this concept", "title" : "a title for the concept"}
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
            'group-similarity-score' : contract_group['group-similarity-score'], # get this from contract_group_info
            'group-complexity-score' : contract_group['group-complexity-score'], # get this from contract_group_info
            'tags' : self.get_tags(doc),
        }

        provisions = {}
        for (_block, _type) in tupleized:
            # Collect provision_group statistics from datastore
            print("get_detail: get the %s provision type" % _type)
            provision_name = get_provision_name_from_file(_type)
            provision_group_info = self.datastore.get_provision_group(provision_name)
            if provision_group_info is not None:
                provisions[provision_name] = {
                    'consensus-percentage' : 0, # computed on the fly
                    "prov-similarity-score" : 0, # computed on the fly
                    "prov-similarity-avg" : provision_group_info['prov-similarity-avg'], # get this from provision_group_info
                    "prov-complexity-score" : 0, # computed on the fly
                    "prov-complexity-avg" : provision_group_info['prov-complexity-avg'], # get this from provision_group_info
                    "provision-tag" : "some-label", # computed on the fly
                }
            else: 
                # TODO: log an error here and 
                # raise an error about not having data for this
                provisions[provision_name] = {}
        document['provisions'] = provisions
        document['concepts'] = self.get_concept_detail()
        return document

def testing():
    """ test that the class is working """
    schema = AgreementSchema()
    print("loading the nondisclosure schema...")
    schema.load_schema("nondisclosure.ini")
    provisions = schema.get_provisions()
    print(provisions)
    print("\n\n")
    print("Do those provisions look right?\n") 

    a = Alignment(schema=schema)
    content = "Confidential Information/Disclosing Party/Receiving Party. Confidential Information is stuff that really really matters."
    print("Test things on a 'long' paragraph.")
    toks = a.tokenize(content)
    result = a.align(toks)
    #print(result)
    markup = a.get_markup(result)
    print(markup)
    print("\nOutput the JSON details\n")

    import json
    print(json.dumps(a.get_detail(result)))

    """ example of some simple chunking """ 
    #for sent in testset: 
    #    for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
    #        if hasattr(chunk, 'label'):
    #            print(chunk.label(), ' '.join(c[0] for c in chunk.leaves()))

def testr():
    filename = "nda-0000-0014.txt"
    print("obtain a corpus...")
    from classifier import build_corpus
    corpus = build_corpus()

    schema = AgreementSchema()
    schema.load_schema('nondisclosure')
    aligner = Alignment(schema=schema, vectorizer=TFIDF_VECT, all=True)
    doc = corpus.raw(filename)
    paras = aligner.tokenize(doc)
    aligned_provisions = aligner.align(paras) # aligned_provisions is a list of tuples
    results = aligner.get_markup(aligned_provisions)
    print(results)
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

"""
Bypass main
"""
if __name__ == "__main__":
    pass