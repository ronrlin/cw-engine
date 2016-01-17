#!/usr/bin/python
import configparser
import os
import re

"""
structure.py provides utilities and services for ingesting and accessing 
training files and schema.  Schema are configuration files that encode 
information about a kind of an agreement.  Training files compile sample 
provisions from agreements into a single data source.


"""

BASE_PATH = "./"
SCHEMA_PATH = os.path.join(BASE_PATH, "schema/")
TRAIN_PATH = os.path.join(BASE_PATH, "train/")

class AgreementSchema(object):
	"""
	"""
	def __init__(self):
		pass

	def load_schema(self, schema_type):
		"""	
		Loads schema information from schema/*.ini. 

		:param schema_type: the schema to load, ie: "convertible_debt.ini"
		"""
		config = configparser.ConfigParser()

		# The name of the schema files should be the same as the agreement_typ
		if (".ini" not in schema_type):
			schema_type = schema_type + ".ini"

		print("A %s agreement is being loaded" % schema_type)

		config.read(SCHEMA_PATH + schema_type)
		sections = config.sections()
		self.version = config['general']['version']
		self.agreement_type = config['general']['agreement_type']
		self.provisions = config.items('provisions')
		self.concepts = config.items('concepts')
		self.tags = config.items('tags')
		self.entities = config.items('entities')

	def get_entities(self):
		"""
		Returns a list of tuples which contain (tag_name, tag_values). 
		:param tag_values: may contain comma-separated values. 
		"""		
		return self.entities

	def get_tags(self):
		"""
		Returns a list of tuples which contain (tag_name, tag_values). 
		:param tag_values: may contain comma-separated values. 
		"""		
		return self.tags

	def get_provisions(self):
		"""
		Returns a list of tuples which contain (name, path_to_training_file).  The 
		name corresponds to the name of a provision.  The path_to_training_file is
		a used for bootstrapping the provision classifier. 
		"""
		return self.provisions

	def get_concepts(self):
		"""	
		Returns a list of tuples which correspond to the concepts expected in this 
		agreement type.  Tuples contain information in form of (provision_source, 
		concepts).  'concepts might be a comma-delimited string.'
		"""
		return self.concepts

	def get_version(self):
		"""	
		Returns the version of the schema being used.
		"""
		return self.version

	def get_agreement_type(self):
		"""	
		Returns a string corresponding to the agreement_type.
		"""
		return self.agreement_type

def load_training_data():
	"""
	Returns a dict of provision_names => provision training file.  
	"""
	train_files = [f for f in os.listdir(TRAIN_PATH) if re.match(r'train_', f)]
	provisions = {}
	for f in train_files:
		provision_name = f[6:] 
		provisions[provision_name] = "train/" + f
	return provisions

def get_provision_name_from_file(filename, dashed=False):
	provisions = load_training_data()
	provision_name = [name for name, fi in provisions.iteritems() if (fi == filename)]
	if not provision_name:
		return "invalid"
	else: 
		provision_name = provision_name[0]
		if (dashed):
			provision_name = provision_name.replace("_", "-")
		return provision_name

def init(): 
	"""
	init loads schema definitions into files.
	"""
	# ##################################
	# Create the convertible debt schema
	config = configparser.ConfigParser()
	config['general'] = {
		'agreement_type': 'convertible_debt',
		'version': '1.0',
	}

	config['provisions'] = {
		'severability': 'train/train_severability',
		'interest_rate': 'train/train_interest_rate',
		'principal_amount': 'train/train_principal_amount',
		'notices': 'train/train_notices_and_notifications',
		'registration_rights': 'train/train_registration_rights',
	}

	config['concepts'] = {
		'interest_rate'	: 'interest_rate',
		'recitals' : 'interest_rate, maturity_date',
		'intro' : 'party, counterparty',
	}

	with open(SCHEMA_PATH + 'convertible_debt.ini', 'w') as configfile:
		config.write(configfile)

	# ##################################
	# Create the nondisclosure agreement schema
	config = configparser.ConfigParser()
	config['general'] = {
		'agreement_type': 'nondisclosure',
		'version': '1.0',
	}

	config['provisions'] = {
		'confidential_information': 'train/train_confidential_information',
		'nonconfidential_information': 'train/train_nonconfidential_information',
		'obligations_receiving_party': 'train/train_obligations_of_receiving_party',
		'time_period': 'train/train_time_period',
		#'term_defined': 'train/train_term_defined',
		#'term_non_defined': 'train/train_term_non_defined',
	}

	config['concepts'] = {
		'time_period': 'term_defined, term_non_defined'		
	}

	config['tags'] = {
		'time_period': 'unilateral, mutual'		
	}

	with open(SCHEMA_PATH + 'nondisclosure.ini', 'w') as configfile:
		config.write(configfile)

def main():
	pass

"""
Bypass main() function when loading module.
"""
if __name__ == "__main__":
    pass