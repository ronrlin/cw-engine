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
from provision import ProvisionMiner

import numpy as np
import math

BASE_PATH = "./"
DATA_PATH = os.path.join(BASE_PATH, "data/")

COUNT_VECT = 1
TFIDF_VECT = 2
staticMode = False

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
        self.tag_dict = None
        self.entity_dict = None
        self.raw_content = None
        provisions = None

        if (not all):
            print("Load %s agreement training provisions" % schema.get_agreement_type())
            provisions = schema.get_provisions()
        else:
            # to load all files
            provisions = load_training_data().items()

        # provisions is a tuple (provision_name, provision_path)
        training_file_names = [p[1] for p in provisions]
        #provision_names = [p[0] for p in provisions]

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
            self.vectorizer = CountVectorizer(input='content', stop_words=None, ngram_range=(1,3))
            self.vectorizer2 = CountVectorizer(input='content', stop_words=None, ngram_range=(1,3))
        elif (vectorizer == TFIDF_VECT):
            self.vectorizer = TfidfVectorizer(input='content', stop_words=None, max_df=1.0, min_df=1, ngram_range=(1,3))
            self.vectorizer2 = TfidfVectorizer(input='content', stop_words=None, max_df=1.0, min_df=1, ngram_range=(1,3))

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

    # TODO: REMOVE VERSION from aligncore  --- it just uses a diff't tokenization scheme. use blankline tokenizer
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
        # Inspect the features of tupleized
        # tags
        self.build_tag_dict(tupleized)
        self.build_entities_dict(tupleized)

        # TODO: this might possibly be loaded when Alignment() is instantiated.
        feature = Feature()
        feature.load_trainers()
        # feature.load_dict()

        # TODO: you should pass tupleized into text_identify
        self.provision_features = feature.text_identify(content)
        
        # ContractGenome Fit goes here?
        # TODO: Based on the tupleized classification, the feature classification,         
        #print("about to do sanity check")
        #print([b for (a,b) in self.provision_features])
        #tupleized = self.sanity_check(tupleized)
        return tupleized

    def sanity_check(self, tupleized):
        """ Suppresses bad classification """ 
        # TODO: suppress output if a provision is "nonconsensus" (<20%?)
        #provision_features = [("some text", 'features/feature_title_paragraph'),("some text", 'features/feature_title_paragraph', "some text")]
        provision_features = self.provision_features
        new_tupleized = []
        for i, (provision, _type) in enumerate(tupleized):
            feature_type = provision_features[i][1]
            #self.provisionstats[_type]
            if ("title_paragraph" in feature_type): 
                new_tupleized.append((provision, ""))
            elif ("signature_line" in feature_type):
                new_tupleized.append((provision, ""))
            elif ("definitions" in feature_type):
                new_tupleized.append((provision, _type))
            elif ("normal_text" in feature_type):
                new_tupleized.append((provision, _type))
            elif ("preamble" in feature_type):
                new_tupleized.append((provision, _type))
            elif ("recital" in feature_type):
                #if "recital" in _type:
                new_tupleized.append((provision, _type))
                #else:
                #    new_tupleized.append((provision, _type))
            else:
                new_tupleized.append((provision, _type))
        return new_tupleized

    def tokenize(self, content):
        """ 
        :param content: is a string that is to be tokenized
        return list of strings
        """
        self.raw_content = content
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        return tokenizer.tokenize(content)

    def get_new_alt_text(self, provision_type, text, increment=0):
        if increment > 0:
            new_text = ""
        else:
            agreement_type = self.schema.get_agreement_type()
            pm = ProvisionMiner()
            new_text = pm.find_better(provision_type, agreement_type)

        new_text_block = "<span id='provision-" + get_provision_name_from_file(provision_type, True) + "-" + str(increment) + "' class='provision " + get_provision_name_from_file(provision_type, True) + "'>" + new_text + "</span>"
        alt_text = "<div id='provision-" + get_provision_name_from_file(provision_type, True) + "-" + str(increment) + "' class='provision " + get_provision_name_from_file(provision_type, True) + "'>" + "<span class='strikethrough'>" + text + "</span>" + " " + new_text_block + "</div>"
        return alt_text

    def get_alt_text(self, provision_type, text, increment=0):
        """ Get the alternate and redlined text. """
        new_text = "There will be new text here."
        print("type of provision is %s" % provision_type)

        if increment > 0:
            new_text = ""
        else: 
            if provision_type.replace("train/train_", "") == "confidential_information":
                new_text = """'Confidential Information' means (whether disclosed directly or indirectly, in writing, electronically, orally, or by inspection or viewing, or in any other form or medium) all proprietary, non-public information of or relating to the Disclosing Party or any of its Affiliates, including but not limited to, financial information, customer lists, supplier lists, business forecasts, software, sales, merchandising and marketing plans and materials, proprietary technology and products, whether or not subject to registration, patent filing or copyright, and all notes, summaries, reports, analyses, compilations, studies and interpretations of any Confidential Information or incorporating any Confidential Information, whether prepared by or on behalf of the Disclosing Party or the Receiving Party.  Confidential Information shall also include the fact that discussions or negotiations are taking place concerning the Transaction between the Disclosing Party and the Receiving Party, and any of the terms, conditions or others facts with respect to any such Transaction, including the status thereof."""
            elif provision_type.replace("train/train_", "") == "nonconfidential_information":
                new_text = """The provisions of this Agreement shall not apply to any Confidential Information which:
    (a) (i) was already known to or in the possession of the Recipient prior to its disclosure pursuant to this Agreement, (ii) was disclosed to the Recipient by a third party not known by Recipient to be under a duty of confidentiality to the Discloser or (iii) which Recipient can establish by competent documentation was independently developed by the Recipient; or
    (b) is now or hereafter comes into the public domain through no violation of this Agreement by the Recipient; or
    (c) is requested or required by a subpoena or other legal process served upon or otherwise affecting the Recipient.  In such event, the Recipient shall, to the extent permitted by law, notify the Discloser as promptly as is practicable, and the Recipient shall use commercially reasonable efforts to cooperate with the Discloser, at the Discloser's sole cost and expense, in any lawful effort to contest the validity of such subpoena or legal process.  Notwithstanding the foregoing, the Recipient may, without giving notice to the Discloser, disclose Confidential Information to any governmental agency or regulatory body having or claiming to have authority to regulate or oversee any aspect of the Recipient's business or the business of the Recipient's affiliates or representatives; or
    (d) the extent necessary or appropriate to effect or preserve Bank of America's security (if any) for any obligation due to Bank of America from Company or to enforce any right or remedy or in connection with any claims asserted by or against Bank of America or any of its Representatives or the Company or any other person or entity involved in the Transaction.
    """
            elif provision_type.replace("train/train_", "") == "obligation_of_receiving_party":
                new_text = """COUNTERPARTY and COMPANY mutually agree to hold each other's Proprietary Information in strict confidence, not to disclose such Proprietary Information to any third parties without the written permission of the Disclosing Party, and not to use the other party's Proprietary Information for its own purposes or for any reason other than for the Purpose.  Other uses are not contemplated and are strictly prohibited; except that, subject to Section 1, Receiving Party may disclose the Disclosing Party's Proprietary Information only if the Receiving Party is required by law to make any such disclosure that is prohibited or otherwise constrained by this Agreement, provided that the Receiving Party will, to the extent legally permissible, provide the Disclosing Party with prompt written notice of such requirement so that the Disclosing Party may seek, at its own expense, a protective order or other appropriate relief.  Subject to the foregoing sentence, such Receiving Party may furnish only that portion of the Proprietary Information that the Receiving Party is legally compelled or is otherwise legally required to disclose; provided, further, that the Receiving Party shall provide such assistance as the Disclosing Party may reasonably request in obtaining such order or other relief."""
            elif provision_type.replace("train/train_", "") == "waiver":
                new_text = """Waiver.  No waiver of any provisions or of any right or remedy hereunder shall be effective unless in writing and signed by both party's authorized representatives.  No delay in exercising, or no partial exercise of any right or remedy hereunder, shall constitute a waiver of any right or remedy, or future exercise thereof, nor shall any such delay or partial exercise change the character of the Confidential Information as such."""
            elif provision_type.replace("train/train_", "") == "severability":
                new_text = """Severability.  If any provision of this Agreement is held to be illegal, invalid or unenforceable under present or future laws effective during the term hereof, such provisions shall be fully severable; this Agreement shall be construed and enforced as if such severed provisions had never comprised a part hereof; and the remaining provisions of this Agreement shall remain in full force and effect and shall not be affected by the severed provision or by its severance from this Agreement."""
            elif provision_type.replace("train/train_", "") == "integration":
                new_text = """This Agreement supersedes in full all prior discussions and agreements between the Parties relating to the Confidential Information, constitutes the entire agreement between the Parties relating to the Confidential Information, and may be amended, modified or supplemented only by a written document signed by an authorized representative of each Party. """
            else:
                new_text = ""

        agreement_type = self.schema.get_agreement_type()
        new_text_block = "<span id='provision-" + get_provision_name_from_file(provision_type, True) + "-" + str(increment) + "' class='provision " + get_provision_name_from_file(provision_type, True) + "'>" + new_text + "</span>"
        alt_text = "<div id='provision-" + get_provision_name_from_file(provision_type, True) + "-" + str(increment) + "' class='provision " + get_provision_name_from_file(provision_type, True) + "'>" + "<span class='strikethrough'>" + text + "</span>" + " " + new_text_block + "</div>"
        return alt_text

    def get_markup(self, tupleized, provisionstats, redline=False):
        """ returns content with markup to identify provisions within agreement 

            tupleized is an array of tuples [(_type, text), (_type, text), ..]

            provisionstats is a dict of key=>value pairs, s.t. 
                provisionstats['_type'] = { key : value, .. }
                
                key/values stored in provisionstats:
                    'provision-readable' : 'name of the provision',
                    'consensus-percentage' : ''
                    "prov-similarity-score" : ''
                    "prov-similarity-avg" : ''
                    "prov-complexity-score" : ''
                    "prov-complexity-avg" : ''
                    "contractwiser-score" : ''

        """
        _markup_list = []
        inc = dict((y,0) for (x, y) in tupleized)
        for (_block, _type) in tupleized:
            text = _block
            if not _type:
                text = "<p><div>" + text + "</div></p>"
            else: 
                provision_name = get_provision_name_from_file(_type, dashed=True)
                if provision_name in provisionstats.keys():
                    sim_score = provisionstats[provision_name]["prov-similarity-score"]
                    sim_mean = provisionstats[provision_name]["similarity"]["mean"]
                    sim_std = math.sqrt(provisionstats[provision_name]["similarity"]["var"])

                    gulpease = provisionstats[provision_name]["prov-gulpease"]
                    gulpease_mean = provisionstats[provision_name]["gulpease"]["mean"]
                    gulpease_std = math.sqrt(provisionstats[provision_name]["gulpease"]["var"])

                    flesch = provisionstats[provision_name]["prov-flesch"]
                    flesch_mean = provisionstats[provision_name]["flesch"]["mean"]
                    flesch_std = math.sqrt(provisionstats[provision_name]["flesch"]["var"])

                    consensus_score = provisionstats[provision_name]["consensus-percentage"]

                    reqs = [filename for (prov, filename) in self.schema.get_provisions()]

                    if redline:
                        _expected = _type in reqs
                        decision = self.redline_decision(provisionstats[provision_name], _expected)
                        if decision:
                            text = self.get_alt_text(_type, text, inc[_type])
                            #text = self.get_new_alt_text(_type, text, inc[_type])
                        else: 
                            print("decision made to not redline.")
                            text = "<div id='provision-" + get_provision_name_from_file(_type, True) + "-" + str(inc[_type]) + "' class='provision " + get_provision_name_from_file(_type, True) + "'>" + text + "</div>"
                            text = "<p>" + text + "</p>" #TODO: is the p tag necessary here?

                    else:
                        text = "<div id='provision-" + get_provision_name_from_file(_type, True) + "-" + str(inc[_type]) + "' class='provision " + get_provision_name_from_file(_type, True) + "'>" + text + "</div>"
                        text = "<p>" + text + "</p>" #TODO: is the p tag necessary here?

                else: 
                    #TODO: this can be consolidated with the case "if not _type" case 
                    text = "<div id='provision-" + get_provision_name_from_file(_type, True) + "-" + str(inc[_type]) + "' class='provision " + get_provision_name_from_file(_type, True) + "'>" + text + "</div>"
                    text = "<p>" + text + "</p>" #TODO: is the p tag necessary here?

            _markup_list.append(text)
            inc[_type] = inc[_type] + 1
        return " ".join(_markup_list)

    def redline_decision(self, stats, expected):
        _consensus = stats["consensus-percentage"]

        sim_score = stats["prov-similarity-score"]
        sim_mean = stats["similarity"]["mean"]
        sim_std = math.sqrt(stats["similarity"]["var"])
        _similarity = (sim_score - sim_mean) / sim_std

        gulpease = stats["prov-gulpease"]
        gulpease_mean = stats["gulpease"]["mean"]
        gulpease_std = math.sqrt(stats["gulpease"]["var"])

        flesch = stats["prov-flesch"]
        flesch_mean = stats["flesch"]["mean"]
        flesch_std = math.sqrt(stats["flesch"]["var"])
        _complexity = (flesch - flesch_mean) / flesch_std

        flesch = stats["prov-flesch"]
        flesch_mean = stats["flesch"]["mean"]
        flesch_std = math.sqrt(stats["flesch"]["var"])
        _variability = 1 / (flesch_std * flesch_std)

        decision = np.array([expected, _consensus, _similarity, _complexity, _variability])
        
        _expected_weight = 0.08
        _consensus_weight = 0.15
        _similarity_weight = 0.30
        _complexity_weight = 0.22
        _variability_weight = 0.25
        weights = np.array([_expected_weight, _consensus_weight, _similarity_weight, _complexity_weight, _variability_weight])

        # Linear Regression Model to output a Redline Decision
        # Logic for redline

        ## Is the provision required?
        ##### If required/expected
        ##### If not expected, then consider striking. 

        ## Is the provision consensus? 
        ##### If it is less than n% common, then consider striking?
        ##### If it is common...

        ## If the similarity is low:
        ##### 
        ##### 
        ## If the flesch is low: 
        #####
        #####

        #This is crap############
        if _consensus < 70:
            return True
        else:
            return False

        #########################

    def build_entities_dict(self, tupleized):
        entities = self.schema.get_entities()
        output = []

        import ner
        import config
        ner_settings = config.load_ner()
        tagger = ner.SocketNER(host=ner_settings['hostname'], port=int(ner_settings['port']))

        # Let's build up a dictionary of global named entities
        globalnerz = tagger.get_entities(self.raw_content)
        for k in globalnerz.keys():

            if len(globalnerz[k]) > 1:
                globalnerz[k] = list(set(globalnerz[k]))
                # strip out values from the NER which we know to be bad
                for values in globalnerz[k]:
                    if k == "ORGANIZATION":
                        # list of strings that are gonna get stripped away
                        discard = ["the State of California", "State of California","Third Parties", "Residual Information", "Disclosure of Confidential Information", "Disclosing Party to Receiving Party", "Confidential Information to Receiving Party","Disclosing Party", "Receiving Party","Parties", "Company", "Confidential Information", "USA", "Discloser's Confidential Information"]
                        globalnerz[k] = [x for x in globalnerz[k] if x not in discard] 

                    elif k == "PERSON":
                        pass
                    elif k == "DATE":
                        try: 
                            import datetime
                            globalnerz[k] = [x for x in globalnerz[k] if len(str(x)) != 4] 

                        except UnicodeEncodeError:
                            print("gracefully handle error: %s" % globalnerz[k])                        

                if globalnerz[k]:
                    nerz_most_common = max(set(globalnerz[k]), key=globalnerz[k].count)
                    result = {}
                    result['type'] = k + "_most_common"
                    result['category'] = k + "_most_common"
                    result['text'] = nerz_most_common
                    result['reference-info'] = ''
                    #output.append(result)
                #else:
                #    del globalnerz[k]

            # TODO: look here for adding smart
            result = {}
            result['type'] = k
            result['category'] = k
            result['text'] = globalnerz[k]
            result['reference-info'] = ''
            #output.append(result)

        #somehow check the length? 
        for tag in entities:
            # tag is a tuple
            tag_name = tag[0]
            tag_values = tag[1].split(",") # list of all possible values

            search_text = " ".join([_block for (_block, _type) in tupleized if tag_name in _type])
            nerz = tagger.get_entities(search_text)

            for entity_type in tag_values:
                if (nerz.get(entity_type) != None):
                    result = {}
                    result['type'] = tag_name + "_" + entity_type
                    result['category'] = get_tag_text(tag_name + "_" + entity_type)
                    result['reference-info'] = '' #some id into a reference db
                    result['text'] = nerz[entity_type]
                    output.append(result)

            #import nltk 
            #sentences = nltk.sent_tokenize(search_text)
            #tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
            #tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
            #if tagged_sentences:
            #    cardinalnum = [word for sent in tagged_sentences for (word, pos) in sent if pos == "CD"]
            #    cardinalnum = [x for x in cardinalnum if x not in ["(", ")"]]
            #    if cardinalnum:
            #        result = {}
            #        result['type'] = tag_name + "_cardinal_number"
            #        result['category'] = tag_name + "_cardinal_number"
            #        result['reference-info'] = '' #some id into a reference db
            #        result['text'] = cardinalnum
            #        output.append(result)

            import re
            duration_match = []
            for m in re.finditer("\(?\d+\)? (years|months|days|year|month|day)", search_text):       
                duration_match.append(m.group(0))

            if duration_match:
                result = {}
                result['type'] = tag_name + "_duration"
                result['category'] = get_tag_text(tag_name + "_duration")
                result['reference-info'] = '' #some id into a reference db
                result['text'] = duration_match
                output.append(result)

        self.entity_dict = output
        return         

    def build_tag_dict(self, tupleized):
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
                nltk_is_stupid = [ [key] for key in mapped.keys()]
                #print(nltk_is_stupid)
                #tagged_corpus = CategorizedPlaintextCorpusReader(DATA_PATH, mapped.keys(), cat_map=mapped)
                tagged_corpus = CategorizedPlaintextCorpusReader(DATA_PATH, nltk_is_stupid, cat_map=mapped)                
                #vectorizer = TfidfVectorizer(input='content', stop_words=None, ngram_range=(1,2))
                vectorizer = CountVectorizer(input='content', stop_words=None, ngram_range=(1,2))
                #vectorizer = TfidfVectorizer(input='content', stop_words=None, ngram_range=(1,2))
                from classifier import AgreementVectorClassifier 
                classifier = AgreementVectorClassifier(vectorizer, tagged_corpus)
                classifier.fit()
                result['type'] = tag_name
                result['category'] = classifier.classify_data([self.raw_content])
                result['reference-info'] = get_reference_info(tag_name, result['category']) #some id into a reference db
                result['text'] = get_tag_text(tag_name) #TODO: this is how the tags get displayed
                output.append(result)
            else: # marked for removal
                print("problem in get_tag!")
                result['type'] = "ERROR"
                result['category'] = ""
                result['reference-info'] = 'ERROR' #some id into a reference db
                result['text'] = 'ERROR' #TODO: this is how the tags get displayed
                output.append(result)

        self.tag_dict = output
        return 

    def compute_score(self, doc_similarity, group_similarity, doc_complexity, group_complexity):
        points = 0
        # agreement term - 10 points
        # disclosure type - 10 points
        # 0-10 similarity of confidential info
        # 0-10 similarity of document
        # 0-5 simplicity of document
        # 25 for all req'd provisions
        # 0-20 for similarity to the standard
        """
        # search entity_dict for time_period_DATE
        term = [entity for entity in self.entity_dict if entity['type'] == 'time_period_DATE']
        if term[0]['text']:
            points += 11

        # search tag_dict for disclosure_type
        disclosure_type = [tag for tag in self.tag_dict if tag['type'] = 'disclosure_type']
        if disclosure_type[0]['text'] == 'mutual':
            points += 15
        else:
            points += 5

        # check for missing provisions
        provisions_found = set([_type for (_block, _type) in tupleized])
        provisions_expected = set([provision_name for (provision_name, path) in self.schema.get_provisions()])
        missing = set(provisions_expected) - set(provisions_found)
        if not missing: 
            points += 20
        else:
            points += len(provisions_found) * (20 / len(provisions_expected))

        # check for confidential information
        points += 11


        complexity_contrib = 20
        similarity_contrib = 20

        """
        # compare tupleized with self.schema.get_provisions()
        # calculate similarity with a standard
        score = ((1 - doc_complexity/group_complexity) + (doc_similarity/group_similarity))/2 * 100
        return round(score, 1)

    def calc_provisionstats(self, tupleized):
        import statistics
        import provision
 
        contract_group = self.datastore.get_contract_group(self.schema.get_agreement_type()) 

        astats = statistics.AgreementStatistics(tupleized)
        astats.transform()
        aparams = astats.calculate_stats()

        doc = " ".join([e[0] for e in tupleized])
        
        docstats = {}
        docstats["doc-complexity-score"] = round(aparams["doc_gulpease"], 1)
        docstats["group-complexity-score"] = contract_group["gulpease"]["mean"]
        docstats["group-complexity-std"] = math.sqrt(contract_group["gulpease"]["var"])
        docstats["group-complexity-min"] = contract_group["gulpease"]["min"]
        docstats["group-complexity-max"] = contract_group["gulpease"]["max"]

        docstats["doc-flesch-score"] = round(aparams["doc_flesch"], 1)
        docstats["group-flesch-score"] = contract_group["flesch"]["mean"]
        docstats["group-flesch-std"] = math.sqrt(contract_group["flesch"]["var"])
        docstats["group-flesch-min"] = contract_group["flesch"]["min"]
        docstats["group-flesch-max"] = contract_group["flesch"]["max"]

        docstats["doc-word-count"] = aparams["word_count"]
        docstats["group-length-mean"] = contract_group["word_count"]["mean"]
        docstats["group-length-std"] = math.sqrt(contract_group["word_count"]["var"])
        docstats["group-length-min"] = contract_group["word_count"]["min"]
        docstats["group-length-max"] = contract_group["word_count"]["max"]

        docstats["doc-syllable-count"] = aparams["syllable_count"]
        docstats["group-syllable-mean"] = contract_group["syllable_count"]["mean"]
        docstats["group-syllable-std"] = math.sqrt(contract_group["syllable_count"]["var"])
        docstats["group-syllable-min"] = contract_group["syllable_count"]["min"]
        docstats["group-syllable-max"] = contract_group["syllable_count"]["max"]

        corpus_stats = statistics.CorpusStatistics(self.agreement_corpus)
        docstats["doc-similarity-score"] = round(corpus_stats.calculate_similarity(doc)['similarity']['mean'], 1)
        docstats["group-similarity-score"] = contract_group["similarity"]["mean"]
        docstats["group-similarity-std"] = math.sqrt(contract_group["similarity"]["var"])
        docstats["group-similarity-min"] = contract_group["similarity"]["min"]
        docstats["group-similarity-max"] = contract_group["similarity"]["max"]

        provisionstats = {}
        print("scroll through tupleized to generate provisionstats")
        for (_block, _type) in tupleized:
            # Collect provision_group statistics from datastore
            provision_mach_name = get_provision_name_from_file(_type, dashed=False)
            provision_name = get_provision_name_from_file(_type, dashed=True)
            provision_group_info = self.datastore.get_provision_group(provision_mach_name)
            if provision_group_info is not None:
                pstats = statistics.ProvisionStatistics(provision_mach_name)
                provisionstats[provision_name] = {
                    'provision-readable' : provision_name,
                    'provision-description' : provision.get_description(provision_mach_name),
                    'consensus-percentage' : statistics.get_consensus(self.agreement_corpus, _type),
                    "prov-similarity-score" : round(pstats.calculate_similarity(_block)['similarity']['mean'], 1), 
                    "prov-gulpease" : round(astats.get_gulpease(provision_mach_name), 1),
                    "prov-flesch" : round(astats.get_flesch(provision_mach_name), 1),
                    "prov-word-count" : astats.get_word_count(provision_mach_name),
                    "prov-syllable-count" : astats.get_syllable_count(provision_mach_name),
                    "contractwiser-score" : 70.0,
                    "provision-tag" : "Placeholder for information.",
                }
                try:
                    del provision_group_info['_id']
                except KeyError:
                    pass
                provisionstats[provision_name].update(provision_group_info)
                
        return (docstats, provisionstats)

    def insert_entity_markup(self, text):
        """ :text: is a list of strings """ 
        result_text = []
        for paragraph in text:
            for entity in self.entity_dict:
                for search_text in entity['text']:
                    #TODO: replace with spans for theming
                    #replace_text = "<span>" + entity['text'] + "</span>"
                    replace_text = "<u>" + search_text + "</u>"
                    paragraph = paragraph.replace(search_text, replace_text)
            result_text.append(paragraph)
        return result_text

    def get_detail(self, tupleized, redline=False):
        (docstats, provisionstats) = self.calc_provisionstats(tupleized)

        doc = " ".join([e[0] for e in tupleized])

        from statistics import CorpusStatistics
        cstats = CorpusStatistics(self.agreement_corpus)
        # TODO: wouldn't it be great to link a user to similar files?  YES!  Obtain urls to similar_files.
        similar_files = list(cstats.most_similar(doc))
        newtag = {}
        newtag["type"] = "most_similar"
        newtag["category"] = ", ".join(similar_files)
        newtag["text"] = get_tag_text(newtag["type"])
        newtag["reference-info"] = ""
        #self.tag_dict.append(newtag)

        confidential_info = [_block for (_block, _type) in tupleized if "confidential_information" in _type]
        confidential_info = self.insert_entity_markup(confidential_info)
        
        time_period = [_block for (_block, _type) in tupleized if "time_period" in _type]
        time_period = self.insert_entity_markup(time_period)

        provisions_found = set([_type.replace("train/train_","") for (_block, _type) in tupleized])
        provisions_expected = set([provision_name for (provision_name, path) in self.schema.get_provisions()])
        missing = list(set(provisions_expected) - set(provisions_found))

        document = dict()
        document['mainDoc'] = {
            '_body' : self.get_markup(tupleized, provisionstats, redline),
            'agreement_type' : self.schema.get_agreement_type(), 
            'text-compare-count' : len(self.agreement_corpus.fileids()), 
            'contractwiser-score' : self.compute_score(docstats["doc-similarity-score"], docstats["group-similarity-score"], docstats["doc-complexity-score"], docstats["group-complexity-score"]),
            'tags' : self.tag_dict,
            'entities' : self.entity_dict,
            'summary' : { 
                "confidential-info" : confidential_info,
                "time-period" : time_period,
                "missing-provisions" : missing,
            }, 
        }
        document['mainDoc'].update(docstats)
        document['provisions'] = provisionstats
        document['concepts'] = {}
        return document

def get_tag_text(tag_name):
    """ Eventually, this should be information in a database."""
    if tag_name == "disclosure_type":
        return """When you have to share sensitive information, a mutual NDA makes the obligations the same for both parties. A unilateral NDA suggests that the obligations of your agreement only apply to one party. """
    if tag_name == "most_similar":
        return """Check out popular templates that are most similar to this agreement. """
    elif tag_name == "preamble_DATE":
        return "Effective date"
    elif tag_name == "preamble_ORGANIZATION":
        return "Parties"
    elif tag_name == "time_period_duration":
        return "Agreement Term"
    elif tag_name == "time_period_DATE":
        return "Termination date"
    elif tag_name == "governing_law_jurisdiction_LOCATION":
        return "Jurisdiction"
    elif tag_name == "breach_MONEY":
        return "Damages"
    else:
        return "<" + tag_name + " not found>"

def get_reference_info(tag_type, tag_category):
    """ Eventually, this should be information in a database."""
    if tag_type == "disclosure_type":
        if tag_category == "mutual": 
            return """When you have to share sensitive information, a mutual NDA makes the obligations the same for both parties. """
        else:
            return """When you have to share sensitive information, a mutual NDA makes the obligations the same for both parties. """
    else:
        return "<tag not found>"

def testing(filename="nda-0000-0015.txt", agreement_type="nondisclosure"):
    """ test that the class is working """
    schema = AgreementSchema()
    print("loading the %s schema..." % agreement_type)
    schema.load_schema(agreement_type)
    provisions = schema.get_provisions()
    print("we're looking for...")
    print(provisions)
    a = Alignment(schema=schema)
    from classifier import build_corpus
    corpus = build_corpus()
    doc = corpus.raw(filename)
    print("tokenize raw text into sentences.")
    paras = a.tokenize(doc)
    print("alignment...")
    paras = a.simplify(paras)
    result = a.align(paras)

    print("check on the markup")
    document = a.get_detail(result, redline=True)
    import json
    print(json.dumps(document, sort_keys=True))

    """ example of some simple chunking """ 
    #for sent in testset: 
    #    for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
    #        if hasattr(chunk, 'label'):
    #            print(chunk.label(), ' '.join(c[0] for c in chunk.leaves()))  

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