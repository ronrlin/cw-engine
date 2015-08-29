#!/usr/bin/python
import configparser
import os

BASE_PATH = "./"
SCHEMA_PATH = os.path.join(BASE_PATH, "schema/")

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

		config.read(SCHEMA_PATH + schema_type)
		sections = config.sections()
		self.version = config['general']['version']
		self.agreement_type = config['general']['agreement_type']
		self.provisions = config.items('provisions')
		self.concepts = config.items('concepts')

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