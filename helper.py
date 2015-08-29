#!/usr/bin/python
from pymongo import MongoClient
from bson.objectid import ObjectId
from bson.objectid import ObjectId
import bson

"""
TODO: Create a unique index on 'filename'.
TODO: Check for exceptions in the case that uniqueness is broken.
TODO: Consider more robust exception handling.

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

def clear_db():
   """ Empty the database """
   client = MongoClient('localhost', 27017)
   client['wiser_db'].drop_collection('classified')
   print("drop 'classified' collection...")
   client.drop_database('wiser_db')
   print("drop wiser_db...")
   client.close()

def create_contract_db():
   """ Create the 'contracts' db """
   client = MongoClient('localhost', 27017)
   print(client.database_names())
   db = client['wiser_db']
   collection = db['contracts']
   collection.insert_one({'text' : 'some text', 'agreement_type' : 'nondisclosure'})
   print("created 'contracts' collection...")

def create_contract_group_db():
   """ Create the 'contract_group' db """
   client = MongoClient('localhost', 27017)
   print(client.database_names())
   db = client['wiser_db']
   collection = db['contract_group']
   collection.insert_one({ 'agreement_type' : 'nondisclosure', 'group-similarity-score' : 0 })
   print("created 'contract_group' collection...")

def create_provision_group_db():
   """ Create the 'provision_group' db """
   client = MongoClient('localhost', 27017)
   print(client.database_names())
   db = client['wiser_db']
   collection = db['provision_group']
   collection.insert_one({ 'provision_name' : 'confidential_information', 'prov-similarity-avg' : 0, 'prov-complexity-avg' : 0 })
   collection.insert_one({ 'provision_name' : 'nonconfidential_information', 'prov-similarity-avg' : 0, 'prov-complexity-avg' : 0 })
   collection.insert_one({ 'provision_name' : 'obligation_receiving_party', 'prov-similarity-avg' : 0, 'prov-complexity-avg' : 0 })
   print("created 'provision_group' collection...")

def create_db():
   """ Create the 'classified' db, which stores metainformation about analyzed agreements """
   client = MongoClient('localhost', 27017)
   print(client.database_names())
   db = client['wiser_db']
   print("created wiser_db...")
   collection = db['classified']
   print("created 'classified' collection...")
   #collection.createIndex( { 'filename': "hashed" } )
   import csv
   agreements = list()

   # this file is not stored in github
   with open('./train/master-classifier.csv', 'r') as csvfile:
      spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
      for row in spamreader:
         agree = {}
         agree['filename'] = row[0]
         agree['category'] = row[1] 
         agreements.append(agree)

   result = collection.insert_many(agreements)
   print("new records created")
   print(len(result.inserted_ids))
   create_contract_db()
   create_contract_group_db()
   create_provision_group_db()
   client.close()

class WiserDatabase(object):
   """ """
   def __init__(self):
      self.client = MongoClient('localhost', 27017)
      self.db = self.client['wiser_db']
      # collection refers to the repository of classified agreements
      self.collection = self.db['classified']
      # contracts is the repository that stores uploaded agreements
      self.contracts = self.db['contracts']
      # contract_group holds information about groups of agreements, like all ndas
      self.contract_group = self.db['contract_group']
      # provision_group holds information about groups of agreements, like all ndas
      self.provision_group = self.db['provision_group']

   def fetch_by_filename(self, filename):
      """
      Returns records (key/value pairs) corresponding to records with a certain value
      for the 'filename' value.

      :param filename: string

      returns Cursor
      """
      result = self.collection.find_one({'filename' : filename})
      return result

   def fetch_by_category(self, category):
      """
      Returns records (key/value pairs) corresponding to records with a certain value
      for the 'category' value.

      :param category: matching a query on all records.

      returns Cursor
      """
      results = self.collection.find({'category' : category})
      return results

   def get_category_names(self):
      """
      Returns a list of the 'category' names.  Categories are different names for 
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
      new_record = { 'text' : text, 'agreement_type' : agreement_type }
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

   def get_contract_group(self, agreement_type):
      """ Return a dict that represents a contract_group and all its calculated properties """
      result = self.contract_group.find_one({ 'agreement_type' : agreement_type })
      return result

   def get_provision_group(self, provision_name):
      """ Return a dict that represents a provision_group and all its calculated properties """
      result = self.provision_group.find_one({ 'provision_name' : provision_name })
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
      print("result is NOT a dict.\n")

   print("look for an invalid contract id")
   result = datastore.get_contract('1')
   print(result)
   if isinstance(result,dict):
      print("result is a dict.\n")
   else:
      print("result is NOT a dict.\n")

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

   print("test fetch_by_category")
   records = datastore.fetch_by_category('nondisclosure')
   for r in records:
      print(r['category'])
      print(r['filename'])

   print("\n\n")
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