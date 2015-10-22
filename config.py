#!/usr/bin/python
from helper import WiserDatabase

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

# --DATABASE SETTINGS
# specify the database name
db_name = "wiser_db"
# specify the hostname
db_hostname = 'localhost'
# specify the port
db_port = 27017
# This file specifies the types of contracts
classifier_source = "./train/master-classifier.csv"
# Tika configuration
tika_hostname = "localhost"
tika_port = 8984

def init():
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
	datastore = WiserDatabase()
	datastore.create_universe()

if __name__ == "__main__":
    init()