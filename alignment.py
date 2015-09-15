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

    def align(self, content, content_id=None):
        """
        Function aligns or classifies sentences passed to the function.

        :param content: a list of strings, where the strings are sentences or paragraphs.

        returns a list of tuples corresponding to the type of provision for each element of the list. 
        """
        # content_id could be an identifier or a path to a file or to the content
        # this might be helpful when you're ready to tag content and calculate meta
        # information.  
        test_vec = self.vectorizer.transform(content)
        results = self.cll.predict(test_vec)
        return list(zip(content, list(results)))

    def concepts(self, tupleized):
        pass

    def continguous_normalize(self, tupleized):
        """
        Function combines contiguous tuples which are the same provision type.  
        This helps simplify the presentation.  

        :param tupleized: is a list of tuples of the form (["string", "of", "text"], type)

        Returns a normalized tupleized.
        """
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
        # Following block creates div statements with custom ids
        inc = dict((y,0) for (x, y) in tupleized)
        for (_block, _type) in tupleized:
            _markup_list.append("<div id='pr-" + get_provision_name_from_file(_type) + "-" + str(inc[_type]) + "' class='provision " + get_provision_name_from_file(_type) + "'>" + "<p>" + _block + "</p>" + "</div>")
            inc[_type] = inc[_type] + 1
        return " ".join(_markup_list)

    def get_tags(self):
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
            result['tag_name'] = tag_name
            result['_value'] = classifier.classify_file("nda-0000-0029.txt")
            output.append(result)
        return output

    def get_detail(self, tupleized):
        # Collect contract_group statistics from datastore
        contract_group = self.datastore.get_contract_group(self.schema.get_agreement_type()) 

        from statistics import AgreementStatistics
        astats = AgreementStatistics(tupleized)
        aparams = astats.calculate_stats()
        # doc is the text of the agreement, formed by joining all the text blocks in the tuple
        doc = [e[0] for e in tupleized]
        doc = " ".join(doc)
        
        # _-----------------
        # TODO:  this after computing 'concepts', so that you can somehow 
        # pass that information to the get_markup() function!!
        # document is the response we will return.
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
            'tags' : self.get_tags(),
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
                # TODO: log an error here
                provisions[provision_name] = {}
        document['provisions'] = provisions
        document['concepts'] = {}
        return document

def test_tag():
    """ 
    !!!This should be removed?!!!
    """ 
    from helper import WiserDatabase 
    datastore = WiserDatabase()
    schema = AgreementSchema()
    print("loading the nondisclosure schema...")
    schema.load_schema("nondisclosure.ini")
    tags = schema.get_tags()
    tupled = []
    """
    This block is responsible for determining what tags are expected for this agreement.
    For each tag, this code obtains the expected values and builds a corpus from tagged 
    agreements.  
    """ 
    # 
    # should we do something about the fileids of a certain type that are not tagged?
    # should the query pass an agreement_type in the query?
    output = []
    for tag in tags:
        # tag is a tuple
        tag_name = tag[0]
        tag_values = tag[1].split(",") # list of all possible values
        for val in tag_values:
            val = val.strip(" ")
            fileids = datastore.fetch_by_classified_tag(tag_name, val)
            thistuple = (zip(fileids, [val] * len(fileids)))
            #need to append elements of thistuple to tupled
            for t in thistuple:
                tupled.append(t)

        print("%s files will be loaded into corpus." % str(len(tupled)))   
        mapped = dict(tupled)
        #print(mapped)
        tagged_corpus = CategorizedPlaintextCorpusReader(DATA_PATH, fileids=mapped.keys(), cat_map=mapped)
        vectorizer = TfidfVectorizer(input='content', stop_words=None, ngram_range=(1,2))
        from classifier import AgreementVectorClassifier 
        classifier = AgreementVectorClassifier(vectorizer, tagged_corpus)
        classifier.fit()
        result = {}
        result['tag_name'] = tag_name
        result['_value'] = classifier.classify_file("nda-0000-0029.txt")
        output.append(result)
    return output

def test_concept():
    from helper import WiserDatabase 
    datastore = WiserDatabase()
    schema = AgreementSchema()
    print("loading the nondisclosure schema...")
    schema.load_schema("nondisclosure.ini")
    concepts = schema.get_concepts()
    tupled = []
    """
    This block is responsible for determining what concepts are expected for this agreement.
    For each concept, the following logic happens:
    - Each concept has a key to a provision and one or more concept_targets.
    - If a concept_target is a path to a trainer, then load the trainers.  
        Pass only the text/provisions that are of the type specified by the key
        Run a train classifier on the loaded trainers, 
        Classify the text
    - If a concept target is a string, then look for a chunk or something

    """ 
    # 
    # should we do something about the fileids of a certain type that are not tagged?
    # should the query pass an agreement_type in the query?
    output = []
    fileids = []
    for c in concepts:
        # concepts is a tuple
        concept_key = c[0] # refers to a provision name
        concept_targets = c[1].split(",") # list of all possible values
        for val in concept_targets:
            path_to_trainer = val.strip(" ")
            # check that path_to_trainer is in fact a trainer
            # if it's not a path to file, then we need to do something else
            fileids.append(path_to_trainer)

        print("%s trainers will be loaded." % str(len(fileids)))   
        concept_corpus = PlaintextCorpusReader(DATA_PATH, fileids=mapped.keys())
        # We probably want to scroll paragraph by paragraph through the trainers
        # Vectorize each paragraph
        #vectorizer = TfidfVectorizer(input='content', stop_words=None, ngram_range=(1,2))
        #from classifier import AgreementVectorClassifier 
        #classifier = AgreementVectorClassifier(vectorizer, tagged_corpus)
        #classifier.fit()
        #whatcomesout = classifier.classify_file("nda-0000-0029.txt")
        # TODO: with a concept, might want to add <span> markup to the text blocks.
        result = {}
        result['concept_' + concept_key] = { 'concept-description' : '', 'concept-text' : ''}
        output.append(result)
    #return output

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
    aligner = Alignment(schema=schema, vectorizer=TFIDF_VECT)
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