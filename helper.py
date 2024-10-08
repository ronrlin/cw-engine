#!/usr/bin/python
from pymongo import MongoClient
from bson.objectid import ObjectId
import bson

"""
Bootstrapping the data for running ContractWiser Engine.

1. Run the interpreter at the CLI:

%> cd <cw-engine root>
>> python
>> import helper
>> helper.create_db()

2. That's it!

To refresh the statistics computed for the application, run:

>> helper.clear_meta_info()
>> helper.create_provision_group_db()
>> helper.create_contract_group_db()
>> helper.load_meta_info()

CW relies on pre-computed statistics, so it's necessary to 
initialize the statistics about provision_groups and 
contract_groups.  

>>> import statistics
>>> statistics.compute_classified_stats()
>>> statistics.compute_contract_group_info()
>>> statistics.display_contract_group_info()
>>> statistics.compute_provision_group_info()

About the datasets

db = wiser_db
collections { 
   classified        : "repository of contracts that have been classified"
   contracts         : "repository of contracts that have been uploaded, analyzed"
   contract_group    : "global stats for classified contracts of a certain category"
   provision_group   : "global stats for provisions of a certain category"
}

Data returned from the 'classified' or 'contracts' collections 
returns something that will resemble this below: 

agreements = [
   {
      'filename' : '891daf5deebf3e31b7bb1c2970ac1c507d50818aa4db2dfd0b7e9b344a340202',
      'category' : 'convertible',  
   }, 
   {
      'filename' : '20d22f00ed2f67c68ff0d81975242187eb07a0c7499da0e16e83f11b4c0372a4',
      'category' : 'indenture',        
   }]

"""

import os
CURATED_PATH = os.path.join("./", "parsed/")

def create_universe():
   """ Create all databases """
   create_db()
   create_contract_db()
   create_contract_group_db()
   create_provision_group_db()
   load_meta_info()
   # next step... load computed information.
   import statistics as s
   s.compute_classified_stats() #does this override load_meta_info()
   s.compute_contract_group_info()
   s.compute_provision_group_info()
   load_meta_info() #I'm not sure why this is necessary, but it is... NEED TO DEBUG why this is overwritten

   #TODO: curated_db stuff here!!!!  

def clear_db():
   """ Empty the database """
   import config
   mongo = config.load_mongo()
   client = MongoClient(mongo['hostname'], mongo['port'])
   client[mongo['db_name']].drop_collection('classified')
   print("drop 'classified' collection...")
   client.drop_database(mongo['db_name'])
   print("drop wiser_db...")
   client.close()

def clear_meta_info():
   """  """
   clear_provision_group_db()
   clear_contract_group_db()

def create_contract_db():
   """ Create the 'contracts' db """
   import config
   mongo = config.load_mongo()
   #client = MongoClient('localhost', 27017)
   client = MongoClient(mongo['hostname'], mongo['port'])
   print(client.database_names())
   db = client[mongo['db_name']]
   collection = db['contracts']
   collection.insert_one({'text' : 'some text', 'agreement_type' : 'nondisclosure'})
   print("created 'contracts' collection...")
   client.close()

def create_contract_group_db():
   """ Create the 'contract_group' db """
   import config
   mongo = config.load_mongo()
   client = MongoClient(mongo['hostname'], mongo['port'])
   db = client[mongo['db_name']]
   collection = db['contract_group']

   # create a document for each agreement_type   
   import os
   print("Obtaining all known agreement types from ./schema.")
   agreement_types = [f.replace(".ini", "") for f in os.listdir("./schema")]
   for thistype in agreement_types:
      collection.insert_one({ 'agreement_type' : thistype })
   print("created 'contract_group' collection...")
   client.close()

def clear_contract_group_db():
   """ Empty the contract_group_info collection """
   import config
   mongo = config.load_mongo()
   client = MongoClient(mongo['hostname'], mongo['port'])
   client[mongo['db_name']].drop_collection('contract_group')
   print("drop 'contract_group' collection...")
   client.close()

def create_provision_group_db():
   """ Create the 'provision_group' db """
   import config
   mongo = config.load_mongo()
   #client = MongoClient('localhost', 27017)
   client = MongoClient(mongo['hostname'], mongo['port'])
   db = client[mongo['db_name']]
   collection = db['provision_group']
   #collection.insert_one({ 'provision_name' : 'confidential_information', 'prov-similarity-avg' : 0, 'prov-complexity-avg' : 0 })
   #collection.insert_one({ 'provision_name' : 'nonconfidential_information', 'prov-similarity-avg' : 0, 'prov-complexity-avg' : 0 })
   #collection.insert_one({ 'provision_name' : 'obligation_receiving_party', 'prov-similarity-avg' : 0, 'prov-complexity-avg' : 0 })

   from structure import load_training_data
   provisions_all = load_training_data().items()
   print(provisions_all)
   for provision in provisions_all:
      provision_name = provision[0]
      provision_file = provision[1]
      info = { 'provision_name' : provision_name }
      collection.insert_one(info)
      print("created a provision_group for %s " % provision_name)

   print("completed the 'provision_group' db.")
   client.close()

def create_curated_db_from_dict():
   """ This function loads all the files in parsed/ directory to a curated db. 

      Relies on creating a parsed/ directory in the web root and providing a bunch 
      of json formatted text files. These get loaded into Mongo.
   """
   import config
   mongo = config.load_mongo()
   client = MongoClient(mongo['hostname'], mongo['port'])      
   db = client[mongo['db_name']]
   print("loading the curated provision database...")
   file_count = 0
   for filename in os.listdir(CURATED_PATH):
      with open(os.path.join(CURATED_PATH, filename),'rb') as json_data:
         my_data = json.load(json_data)
      result = db['curated'].insert_many(my_data)
      file_count += 1
   print("(%s) files loaded." % str(file_count))
   print("created and loaded the curated db.")

def clear_curated_db():
   """ Empty the provision_group_info collection """
   import config
   mongo = config.load_mongo()
   #client = MongoClient('localhost', 27017)
   client = MongoClient(mongo['hostname'], mongo['port'])
   client[mongo['db_name']].drop_collection('curated')
   print("drop 'curated' collection...")
   client.close()

def clear_provision_group_db():
   """ Empty the provision_group_info collection """
   import config
   mongo = config.load_mongo()
   #client = MongoClient('localhost', 27017)
   client = MongoClient(mongo['hostname'], mongo['port'])
   client[mongo['db_name']].drop_collection('provision_group')
   print("drop 'provision_group' collection...")
   client.close()

def create_db():
   """ Create the 'classified' db, which stores metainformation about analyzed agreements """
   import config
   mongo = config.load_mongo()
   #client = MongoClient('localhost', 27017)
   client = MongoClient(mongo['hostname'], mongo['port'])
   print(client.database_names())
   db = client[mongo['db_name']]
   print("created db named %s..." % mongo['db_name'])
   collection = db['classified']
   print("created 'classified' collection...")
   collection.create_index([('filename', 1)], unique=True)
   import csv
   agreements = list()

   import config
   classifier_source = config.classifier_source
   with open(classifier_source, 'r') as csvfile:
      spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
      for row in spamreader:
         agree = {}
         agree['filename'] = row[0]
         agree['category'] = row[1] 
         agreements.append(agree)

   result = collection.insert_many(agreements)
   print("%s new records created in classified db" % str(len(result.inserted_ids)))
   client.close()

def load_meta_info():
   """
   Loads some basic meta information into the 'classified' collection.
   """
   datastore = WiserDatabase()
   tag_mutual = {'disclosure_type' : 'mutual'}
   info = datastore.fetch_by_filename('nda-0000-0001.txt')
   datastore.tag_classified(str(info['_id']), tag_mutual)
   info = datastore.fetch_by_filename('nda-0000-0002.txt')
   datastore.tag_classified(str(info['_id']), tag_mutual)
   info = datastore.fetch_by_filename('nda-0000-0007.txt')
   datastore.tag_classified(str(info['_id']), tag_mutual)
   info = datastore.fetch_by_filename('nda-0000-0016.txt')
   datastore.tag_classified(str(info['_id']), tag_mutual)
   info = datastore.fetch_by_filename('nda-0000-0017.txt')
   datastore.tag_classified(str(info['_id']), tag_mutual)
   info = datastore.fetch_by_filename('nda-0000-0018.txt')
   datastore.tag_classified(str(info['_id']), tag_mutual)
   info = datastore.fetch_by_filename('nda-0000-0019.txt')
   datastore.tag_classified(str(info['_id']), tag_mutual)
   info = datastore.fetch_by_filename('nda-0000-0020.txt')
   datastore.tag_classified(str(info['_id']), tag_mutual)
   info = datastore.fetch_by_filename('nda-0000-0021.txt')
   datastore.tag_classified(str(info['_id']), tag_mutual)
   info = datastore.fetch_by_filename('nda-0000-0022.txt')
   datastore.tag_classified(str(info['_id']), tag_mutual)
   info = datastore.fetch_by_filename('nda-0000-0023.txt')
   datastore.tag_classified(str(info['_id']), tag_mutual)
   info = datastore.fetch_by_filename('nda-0000-0024.txt')
   datastore.tag_classified(str(info['_id']), tag_mutual)
   info = datastore.fetch_by_filename('nda-0000-0025.txt')
   datastore.tag_classified(str(info['_id']), tag_mutual)
   info = datastore.fetch_by_filename('nda-0000-0026.txt')
   datastore.tag_classified(str(info['_id']), tag_mutual)
   info = datastore.fetch_by_filename('nda-0000-0027.txt')
   datastore.tag_classified(str(info['_id']), tag_mutual)
   datastore.tag_classified(str(info['_id']), {'company' : 'google'})
   info = datastore.fetch_by_filename('nda-0000-0035.txt')
   datastore.tag_classified(str(info['_id']), tag_mutual)
   info = datastore.fetch_by_filename('nda-0000-0038.txt')
   datastore.tag_classified(str(info['_id']), tag_mutual)
   info = datastore.fetch_by_filename('nda-0000-0041.txt')
   datastore.tag_classified(str(info['_id']), tag_mutual)

   tag_uni = {'disclosure_type' : 'unilateral'}
   info = datastore.fetch_by_filename('nda-0000-0032.txt')
   datastore.tag_classified(str(info['_id']), tag_uni)
   info = datastore.fetch_by_filename('nda-0000-0039.txt')
   datastore.tag_classified(str(info['_id']), tag_uni)
   info = datastore.fetch_by_filename('nda-0000-0040.txt')
   datastore.tag_classified(str(info['_id']), tag_uni)
   info = datastore.fetch_by_filename('nda-0000-0042.txt')
   datastore.tag_classified(str(info['_id']), tag_uni)
   info = datastore.fetch_by_filename('nda-0000-0051.txt')
   datastore.tag_classified(str(info['_id']), tag_uni)
   info = datastore.fetch_by_filename('nda-0000-0052.txt')
   datastore.tag_classified(str(info['_id']), tag_uni)
   info = datastore.fetch_by_filename('nda-0000-0053.txt')
   datastore.tag_classified(str(info['_id']), tag_uni)
   info = datastore.fetch_by_filename('nda-0000-0054.txt')
   datastore.tag_classified(str(info['_id']), tag_uni)
   info = datastore.fetch_by_filename('nda-0000-0055.txt')
   datastore.tag_classified(str(info['_id']), tag_uni)
   info = datastore.fetch_by_filename('nda-0000-0056.txt')
   datastore.tag_classified(str(info['_id']), tag_uni)
   info = datastore.fetch_by_filename('nda-0000-0057.txt')
   datastore.tag_classified(str(info['_id']), tag_uni)
   info = datastore.fetch_by_filename('nda-0000-0058.txt')
   datastore.tag_classified(str(info['_id']), tag_uni)
   info = datastore.fetch_by_filename('nda-0000-0048.txt')
   datastore.tag_classified(str(info['_id']), tag_uni)

   print("loaded the meta data about nondisclosures.")

   tag_nnn = { 'lease_type' : 'nnn_lease'}
   info = datastore.fetch_by_filename('nnn-0000-0000.txt')
   datastore.tag_classified(str(info['_id']), tag_nnn)
   info = datastore.fetch_by_filename('nnn-0000-0001.txt')
   datastore.tag_classified(str(info['_id']), tag_nnn)
   info = datastore.fetch_by_filename('nnn-0000-0002.txt')
   datastore.tag_classified(str(info['_id']), tag_nnn)
   info = datastore.fetch_by_filename('nnn-0000-0003.txt')
   datastore.tag_classified(str(info['_id']), tag_nnn)
   info = datastore.fetch_by_filename('nnn-0000-0004.txt')
   datastore.tag_classified(str(info['_id']), tag_nnn)
   info = datastore.fetch_by_filename('nnn-0000-0005.txt')
   datastore.tag_classified(str(info['_id']), tag_nnn)
   info = datastore.fetch_by_filename('nnn-0000-0006.txt')
   datastore.tag_classified(str(info['_id']), tag_nnn)
   info = datastore.fetch_by_filename('nnn-0000-0007.txt')
   datastore.tag_classified(str(info['_id']), tag_nnn)
   info = datastore.fetch_by_filename('nnn-0000-0008.txt')
   datastore.tag_classified(str(info['_id']), tag_nnn)
   info = datastore.fetch_by_filename('nnn-0000-0011.txt')
   datastore.tag_classified(str(info['_id']), tag_nnn)
   info = datastore.fetch_by_filename('nnn-0000-0013.txt')
   datastore.tag_classified(str(info['_id']), tag_nnn)
   info = datastore.fetch_by_filename('nnn-0000-0014.txt')
   datastore.tag_classified(str(info['_id']), tag_nnn)
   info = datastore.fetch_by_filename('nnn-0000-0015.txt')
   datastore.tag_classified(str(info['_id']), tag_nnn)
   info = datastore.fetch_by_filename('nnn-0000-0016.txt')
   datastore.tag_classified(str(info['_id']), tag_nnn)
   info = datastore.fetch_by_filename('nnn-0000-0017.txt')
   datastore.tag_classified(str(info['_id']), tag_nnn)
   info = datastore.fetch_by_filename('nnn-0000-0018.txt')
   datastore.tag_classified(str(info['_id']), tag_nnn)
   info = datastore.fetch_by_filename('nnn-0000-0019.txt')
   datastore.tag_classified(str(info['_id']), tag_nnn)
   info = datastore.fetch_by_filename('nnn-0000-0020.txt')
   datastore.tag_classified(str(info['_id']), tag_nnn)
   info = datastore.fetch_by_filename('nnn-0000-0021.txt')
   datastore.tag_classified(str(info['_id']), tag_nnn)
   datastore.tag_classified(str(info['_id']), { 'company' : 'lowes' })

   tag_gross = { 'lease_type' : 'gross_lease'}
   info = datastore.fetch_by_filename('nnn-0000-0023.txt')
   datastore.tag_classified(str(info['_id']), tag_gross)

   tag_sublease = { 'lease_type' : 'sublease'}
   info = datastore.fetch_by_filename('nnn-0000-0032.txt')
   datastore.tag_classified(str(info['_id']), tag_sublease)
   info = datastore.fetch_by_filename('nnn-0000-0028.txt')
   datastore.tag_classified(str(info['_id']), tag_sublease)

   print("loaded the meta data about commercial leases.")

class WiserDatabase(object):
   """ """
   def __init__(self):
      import config
      mongo = config.load_mongo()
      self.client = MongoClient(mongo['hostname'], mongo['port'])      
      self.db = self.client[mongo['db_name']]
      # collection refers to the repository of classified agreements
      self.collection = self.db['classified']
      # contracts is the repository that stores uploaded agreements
      self.contracts = self.db['contracts']
      # contract_group holds information about groups of agreements, like all ndas
      self.contract_group = self.db['contract_group']
      # provision_group holds information about groups of agreements, like all ndas
      self.provision_group = self.db['provision_group']
      # curated is a scrubbed provision_db, containing data about provisions
      self.curated = self.db['curated']

   def fetch_curated(self, feature_class="features/feature_normal_text"):
      """ Returns list of dict corresponding to a feature_class type. """
      return self.curated.find({'feature_class' : feature_class}, { '_id' : 0 })#.limit(50)

   def fetch_feature_class_types(self):
      """ Returns all possible feature types in the curated db. """ 
      return self.curated.distinct('feature_class')

   def fetch_provision_type(self, provision_type):
      """ Returns all possible feature types in the curated db. """ 
      return self.curated.find({'first_guess' : provision_type})

   def fetch_provision_types(self, provision_type):
      """ Returns all possible feature types in the curated db. """ 
      return self.curated.distinct('first_guess')

   def update_curated(self, oid, value):
      result = self.curated.update_one({"_id" : ObjectId(oid) }, {"$set" : parameters})
      return result

   def fetch_by_filename(self, filename):
      """ Returns records (key/value pairs) corresponding to records with a certain value
      for the 'filename' value.

      :param filename: string

      returns Cursor
      """
      result = self.collection.find_one({'filename' : filename})
      return result

   def fetch_by_category(self, category):
      """ Returns records (key/value pairs) corresponding to records with a certain value
      for the 'category' value.

      :param category: matching a query on all records.

      returns Cursor
      """
      results = self.collection.find({'category' : category})
      return results

   def fetch_by_classified_tag(self, key, value):
      """ Adds a tag to a record that is classified.
      :param keyvalue: a dict() of key => value pair to search using.
         ie: find({'key' : 'value'})

      Returns a list of fileids.
      """
      results = self.collection.find({ key : value })
      fileids = [record['filename'] for record in results]
      return fileids      

   def fetch_by_tag(self, keyvalue):
      """ Adds a tag to a record that is classified.
      :param keyvalue: a dict() of key => value pair to search using.
         ie: find({'key' : 'value'})

      Returns a list of fileids.
      """
      results = self.collection.find(keyvalue)
      fileids = [record['filename'] for record in results]
      return fileids            

   def fetch_by_contract_tag(self, key, value):
      """ Adds a tag to one of the uploaded contracts.
      :param keyvalue: a dict() of key => value pair to search using.
         ie: find({'key' : 'value'})
      """
      results = self.contracts.find({ key : value })
      return results      

   def get_category_names(self):
      """ Returns a list of the 'category' names.  Categories are different names for 
      the agreements in the ContractWiser repository.  Examples include 'CONVERTIBLE'
      for Convertible Note agreements, etc...

      returns a list
      """
      return self.collection.distinct("category")

   def add_record(self, filename, category):
      """ Add a record to the wiser_db """
      new_record = { 'filename' : filename, 'category' : category }
      result = self.collection.insert_one(new_record)
      print("one (1) new record created: " + result.inserted_id)
      return result.inserted_id

   def update_record(self, filename, parameters):
      """ """
      parameters['filename'] = filename
      result = self.collection.update_one({'filename' : filename}, {"$set" : parameters})
      print("matched_count : %d" % result.matched_count)
      return result

   def get_contract(self, contract_id):
      """ Return a dict for a contract """
      try:
         #obj_id = ObjectId('55e0a296711b77608bdda709')
         #result = { '_id' : obj_id, 'agreement_type' : 'nondisclosure', 'text' : "Some really really really really long text" }
         result = self.contracts.find_one({ '_id' : ObjectId(contract_id) })
         return result
      except bson.errors.InvalidId:      
         result = None

   def save_contract(self, text, agreement_type = None):
      """ Save the text as a contract """
      from datetime import datetime
      new_record = { "created_date": datetime.now(), 'text' : text, 'agreement_type' : agreement_type }
      result = self.contracts.insert_one(new_record)
      return result.inserted_id

   def update_contract(self, contract_id, agreement_type):
      """ 
         Update the contract record
         :param contract_id: string of the contract_id
         :param agreement_type: string representing the new agreement_type

         Returns an UpdateResult object 
      """
      result = self.contracts.update_one({'_id' : ObjectId(contract_id)}, { '$set' : {'agreement_type' : agreement_type}})
      return result

   def set_standard(self, contract_id):
      """ 
         Update the contract record
         :param contract_id: string of the contract_id

         Returns an UpdateResult object 
      """
      result = self.contracts.update_one({'_id' : ObjectId(contract_id)}, { '$set' : {'is_standard' : True}})
      return result

   def unset_standard(self, contract_id):
      """ 
         Update the contract record
         :param contract_id: string of the contract_id

         Returns an UpdateResult object 
      """
      result = self.contracts.update_one({'_id' : ObjectId(contract_id)}, { '$set' : {'is_standard' : False}})
      return result

   def tag_contract(self, contract_id, tag_dict={}):
      result = self.contracts.update_one({'_id' : ObjectId(contract_id)}, { '$set' : tag_dict})
      return result     

   def tag_classified(self, contract_id, tag_dict={}):
      result = self.collection.update_one({'_id' : ObjectId(contract_id)}, { '$set' : tag_dict})
      return result     

   def get_contract_group(self, agreement_type):
      """ Return a dict that represents a contract_group and all its calculated properties """
      result = self.contract_group.find_one({ 'agreement_type' : agreement_type })
      return result

   def update_contract_group(self, agreement_type, info):
      """ Update a dict that represents a contract_group and all its calculated properties """
      result = self.contract_group.update_one({'agreement_type' : agreement_type}, {'$set' : info}, False)
      return result

   def get_provision_group(self, provision_name):
      """ Return a dict that represents a provision_group and all its calculated properties """
      result = self.provision_group.find_one({ 'provision_name' : provision_name })
      return result

   def update_provision_group(self, provision_name, info):
      """ Update a dict that represents a provision_group and all its calculated properties """
      result = self.provision_group.update_one({'provision_name' : provision_name}, {'$set' : info}, True)
      return result

def testing():
   datastore = WiserDatabase()
   
   #obj_id = ObjectId('55e0a296711b77608bdda709')
   #result = { '_id' : obj_id, 'agreement_type' : 'nondisclosure', 'text' : "Some really really really really long text" }

   print("look for a valid contract id")
   result = datastore.get_contract('55e0a296711b77608bdda709')
   print(result)

   if isinstance(result,dict):
      print("result is a dict.\n")
   else:
      print("result is not a dict.\n")

   print("look for an invalid contract id")
   result = datastore.get_contract('1')
   print(result)
   if isinstance(result,dict):
      print("result is a dict.\n")
   else:
      print("result is not a dict.\n")

   print("insert a dummy record.")
   saved_id = datastore.save_contract("This is some really long text that would typically be found in a legal contract.", "nondisclosure")
   print("A record was saved with id %s " % saved_id)

   print("load the dummy record.")
   contract = datastore.get_contract(saved_id)
   print(contract)
   if isinstance(contract,dict):
      print("contract is a dict.\n")
   else:
      print("ERROR - contract is NOT a dict.\n")

   print("test updating a contract")
   result = datastore.update_contract(saved_id, 'convertible_debt')

   print("add a tag to the contract")
   result = datastore.tag_contract(saved_id, {'disclosure_type' : 'mutual'})

   print("load the dummy record.")
   contract = datastore.get_contract(saved_id)
   print(contract)
   if isinstance(contract,dict):
      print("contract is a dict.\n")
   else:
      print("ERROR - contract is NOT a dict.\n")

   print("search with a tag")
   contracts = datastore.fetch_by_contract_tag('disclosure_type', 'mutual')
   contracts = list(contracts)
   print("%s records were returned" % str(len(contracts)))
   #print(list(contracts))
   for r in list(contracts):
      print("%s" % r['agreement_type'])

   print("\n")

   # We are not handling an empty result well.
   print("test fetch_by_category")
   records = datastore.fetch_by_category('nondisclosure')
   records = list(records)
   print("%s records were returned" % str(len(records)))
   if records is not None:
      for r in records:
         print("%s" % r['filename'])

      print("\n")

   from structure import AgreementSchema
   provisioner = AgreementSchema()
   provisioner.load_schema('nondisclosure')
   print(provisioner.get_provisions())

   print("\nretrieve a contract group...")
   contract_group = datastore.get_contract_group('nondisclosure')
   print(contract_group)

   print("\nretrieve the provision groups")
   for (provision_name, trainer_file) in provisioner.get_provisions():
      print(provision_name)
      provision_group_info = datastore.get_provision_group(provision_name)
      print(provision_group_info)

   print("\nretrieve an undefined provision")
   provision_group_info = datastore.get_provision_group('notaprovision')
   if (provision_group_info is not None):
      print("Something is wrong.")
   else:
      print("Just like you expected.")
           
   print("\nthe end.")

def test_provision_group(provision_group='nondisclosure'):
   from structure import AgreementSchema
   provisioner = AgreementSchema()

   provisioner.load_schema(provision_group)
   print("The expected provisions...")
   print(provisioner.get_provisions())

   datastore = WiserDatabase()

   print("Retrieve the provision groups")
   for (provision_name, trainer_file) in provisioner.get_provisions():
      print(provision_name)
      provision_group_info = datastore.get_provision_group(provision_name)
      print(provision_group_info)

 