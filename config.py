#!/usr/bin/python
import helper
import configparser

"""
Instructions on creating the zip file stored on S3
%> cd <app root>
%> tar -czvf data.tar.gz -C data/ .
"""
datasource_url = "https://s3-us-west-2.amazonaws.com/contractwiser-datasource/data.tar.gz"

######################################################
# This should be set to ./data.  
# It should be something else only for testing.
destination_directory = "./data"
######################################################

# --CSV OF CLASSIFIED FILES
# This file specifies the types of contracts
classifier_source = "./train/master-classifier.csv"

def retrieve_and_load_data():
	""" Run init to set things up. """
	print("Installing ContractWiser engine...")
	print("Fetching datasource from Amazon S3...")
	import urllib
	dest_filename = "./datasource.tar.gz"
	try:
		name, hdrs = urllib.urlretrieve(datasource_url, dest_filename)
		print("Archive is saved locally as %s" % name)
		print(hdrs)
	except IOError, e:
		print(e)
	# Go to AWS and download a zip file of data
	# Extract the AWS zip file
	# Get the classified.csv to load the database
	import zipfile
	import tarfile
	try:
		#change this to be dest_filename
		if tarfile.is_tarfile("./datasource.tar.gz"):
			print("Source file is a tarfile.")
		else:
			print("Source file is not a tarfile.")
			print("exiting with an error")
			return

		t = tarfile.open("./datasource.tar.gz")
		print("Extracting files to %s ..." % destination_directory)
		t.extractall(destination_directory)
	except tarfile.TarError, e:
		print("there was a tarfile problem")
		return

	# TODO: Should we cp the master-classifier.csv somewhere?
	print("Create the universe.")
	print("Database will be called %s." % db_name)
	helper.create_universe()

	def __init__(self):
		pass

def load_settings_file():
	config = configparser.ConfigParser()
	config.read("./settings.ini")
	return config

def load_tika():
	config = load_settings_file()
	params = dict()
	params['hostname'] = config['tika']['hostname']
	params['port'] = config['tika']['port']
	return params

def load_mongo():
	"""	Loads configuration settings from settings.ini """
	config = load_settings_file()
	mongo_params = dict()
	mongo_params['db_name'] = config['mongo']['db_name']
	mongo_params['hostname'] = config['mongo']['hostname']
	mongo_params['port'] = int(config['mongo']['port'])
	return mongo_params

def load_mysql(self):
	config = load_settings_file()
	mysql_params = dict()
	mysql_params['db_name'] = config['mysql']['db_name']
	mysql_params['host'] = config['mysql']['host']
	mysql_params['user'] = config['mysql']['user']
	mysql_params['passwd'] = config['mysql']['passwd']
	return mysql_params


if __name__ == "__main__":
    pass