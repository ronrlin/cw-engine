#!/usr/bin/python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from sklearn import svm

import nltk
from nltk.tokenize import word_tokenize
import os
from structure import AgreementSchema

BASE_PATH = "./"
COUNT_VECT = 1
TFIDF_VECT = 2

"""
Alignment describes the process by which agreements of the same kind of are compared
to one another in a way that "aligns" the provisions within them.  Aligning agreements
is critical to being able to calculate relative frequency of certain provisions.  

"""
class Alignment(object):

    def __init__(self, schema=None, vectorizer=COUNT_VECT, stop_words=None):
        """
        Create an Alignment object. 

        :param schema: specify the AgremeentSchema to use for alignment.
        :param vectorizer: specify the type of vectorization method.
        :param stop_words: specify stop words to drop from texts.
        """
        self.schema = schema
        print("Load %s agreement training provisions" % schema.get_agreement_type())
        provisions_reqd = schema.get_provisions()
        # provisions_reqd is a tuple (provision_name, provision_path)
        training_file_names = [p[1] for p in provisions_reqd]
        provision_names = [p[0] for p in provisions_reqd]       

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
        self.agreement_corpus = PlaintextCorpusReader(BASE_PATH, fileids)
        print("Agreement Corpus of type %s has been loaded." % schema.get_agreement_type())

        if (vectorizer == COUNT_VECT): 
            self.vectorizer = CountVectorizer(input='content', stop_words=None, ngram_range=(1,1))
        elif (vectorizer == TFIDF_VECT):
            self.vectorizer = TfidfVectorizer(input='content', stop_words=None, ngram_range=(1,1))

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
        target2 = list()

        for tfile in self.training_corpus.fileids():
            for tpara in self.training_corpus.sents(tfile):  
                target.append(tfile)
        end_time = time.time()
        print("Time to assemble a target vector is %s seconds" % (end_time - start_time))

        new_target = list()
        for t in target:
            for (name, filename) in provisions_reqd:
                if filename == t:
                    new_target.append(name)
        print(new_target)


        start_time = time.time()
        self.cll = svm.LinearSVC(class_weight='auto')
        self.cll.fit(train_vec, new_target)
        end_time = time.time()
        print("Time to build classifier and fit is %s seconds" % (end_time - start_time))
        print("\nReady for alignment!")

    def align(self, content):
        """
        Function aligns or classifies sentences passed to the function.

        :param content: a list of strings, where the strings are sentences or paragraphs.

        returns a list of tuples corresponding to the type of provision for each element of the list. 
        """
        test_vec = self.vectorizer.transform(content)
        results = self.cll.predict(test_vec)
        return list(zip(content, list(results)))

    def continguous_normalize(self, tupleized):
        provision_types = [e[1] for e in tupleized]
        provision_text = [e[0] for e in tupleized]
        contig = []
        conttypes = []
        returnlist = []
        maxlen = len(provision_types)
        contigctr = 0
        contig.append(contigctr)
        conttypes.append(provision_types[0])
        for i, val in enumerate(provision_types):
            if (i < maxlen - 1):
                if (val == provision_types[i+1]):
                    #they are contig
                    pass
                else: 
                    contigctr += 1
                    contig.append(contigctr)
                    conttypes.append(provision_types[i+1])

        trackr = dict()
        for p in provision_types:
            trackr[p] = list()

        for i, thistype in enumerate(provision_types):
            trackr[thistype].append(i)

        m=[]
        ll = []
        newdict = (trackr)
        nn = set(conttypes)
        for n in nn:
            print(n)
            ids = trackr[n]
            for i in ids:
                if not ll: #if ll is empty
                    ll.append(i)
                else: 
                    if i == ll[-1]+1:
                        ll.append(i)
                    else:
                        m.append(ll)
                        ll = []
                        ll.append(i)        
            if len(ids) == 1 or ll:
                m.append(ll)
                newdict[n] = m
                ll = []
                m = []
        #{'nondisclosure': [[3, 4, 5], [7, 8]], 'intro': [[0, 1, 2]], 'confidential': [[6]]}   
        ll = []
        ctr = 0
        for nn in conttypes:
            print("let's sort %s " % nn)        
            combolists = newdict[nn]
            comboprovs = combolists.pop(0)
            combo = ""
            for c in comboprovs:
                combo = combo + " " + provision_text[c]
            this_tuple = (combo, nn)
            ll.append(this_tuple)
        return ll

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
        for (_block, _type) in tupleized:
            _markup_list.append("<span class='" + _type + "'>" + _block + "</span>")

        _content = "</p><p>".join(_markup_list)
        return "<p>" + _content + "</p>"

    def get_detail(self, tupleized):
        # Collect contract_group statistics from datastore
        contract_group_info = self.datastore.get_contract_group(self.schema.get_agreement_type()) 
        detail = dict()
        detail['body'] = self.get_markup(tupleized)
        detail['agreement_type'] = self.schema.get_agreement_type() # get this from contract_group_info
        detail['text-compare-count'] = len(self.agreement_corpus.fileids()) # get this from contract_group_info
        detail['group-similarity-score'] = 0 # get this from contract_group_info
        for (_block, _type) in tupleized:
            # Collect provision_group statistics from datastore
            print("get_detail: get the %s provision type" % _type)
            provision_group_info = self.datastore.get_provision_group(_type)
            if provision_group_info is not None:
                detail[_type] = {
                    'consensus-percentage' : 0, # computed on the fly
                    "prov-similarity-score" : 0, # computed on the fly
                    "prov-similarity-avg" : provision_group_info['prov-similarity-avg'], # get this from provision_group_info
                    "prov-complexity-score" : 0, # computed on the fly
                    "prov-complexity-avg" : provision_group_info['prov-complexity-avg'], # get this from provision_group_info
                    "provision-tag" : "some-label", # computed on the fly
                }
            else: 
                detail[_type] = {}
        return detail

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
    aligner = Alignment(schema=schema)
    doc = corpus.raw(filename)
    paras = aligner.tokenize(doc)
    aligned_provisions = aligner.align(paras) # aligned_provisions is a list of tuples
    
    tupleized = aligner.continguous_normalize(aligned_provisions)
    print(tupleized)
    return tupleized



"""
Bypass main
"""
if __name__ == "__main__":
    pass