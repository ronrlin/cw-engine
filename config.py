#!/usr/bin/python
from helper import WiserDatabase

#from pdfminer.pdfparser import PDFParser
#from pdfminer.pdfdocument import PDFDocument
#from pdfminer.pdfpage import PDFPage
from pdfminer.pdfpage import PDFTextExtractionNotAllowed
#from pdfminer.pdfinterp import PDFResourceManager
#from pdfminer.pdfinterp import PDFPageInterpreter
#from pdfminer.pdfdevice import PDFDevice

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

def init():
	""" Run init to set things up. """
	print("Installing ContractWiser engine...")
	print("Fetching datasource from Amazon S3...")
	import urllib
	dest_filename = "./datasource.tar.gz"
	try:
		name, hdrs = urllib.urlretrieve(datasource_url, dest_filename)
		print("Archive is saved localled as %s" % name)
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

		t = tarfile.open("./data.tar.gz")
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

def ingest(src_dir="/home/obironkenobi/Projects/sample-dir", src="sample.csv", move_to_data=False, archive=False):
	# input should be a csv like the master-classifier (filename, type)
	print("change the working directory to %s" % src_dir)
	import os
	import zipfile
	import shutil
	import csv

	os.chdir(src_dir)
	agreements = []
	classifier_source = src
	with open(classifier_source, 'r') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
		for row in spamreader:
			agree = {}
			agree['filename'] = row[0].replace(" ", "")
			agree['category'] = row[1].replace(" ", "")
			# row[2] is a key=value pair. 
			agreements.append(agree)

	if archive:
		zipf = zipfile.ZipFile('datasource.zip', 'w')

	for doc in agreements:
		# extract text from pdf
		docname = doc['filename']
		newname = ""
		if (".pdf" in docname):
			newname = doc['category'] + "-" + docname.replace(".pdf", ".txt") 
			newname = newname.replace(" ", "")
			print("handling a pdf... %s will be created." % newname)
			pdf2txt(docname, newname)

		elif (".txt" in docname):
			pass
		elif (".doc" in docname or ".docx" in docname):
			newname = doc['category'] + "-" + docname.replace(".docx", ".txt") 
			newname = newname.replace(" ", "")
			print("handling a docx... %s will be created." % newname)
			doc2txt(docname, newname)
		else: 
			raise PDFTextExtractionNotAllowed		

		if move_to_data:
			print("moving the resulting file to data folder.")
			shutil.move(newname, "/home/obironkenobi/Projects/cw-engine/data/")
		if archive: 
			zipf.write(newname)

	if archive: 
		zipf.close()

	print("it would be nice to upload this to S3...")
	# upload the tar to S3

# pip install python-docx
def doc2txt(filename, txtfile):
	from docx import Document
	document = Document(filename)
	paras = [para.text.encode("utf-8") for para in document.paragraphs]
	text = "\n".join(paras)
	text_file = open(txtfile, "w")
	text_file.write(text)
	text_file.close()
	return 

# Consider custom routine as well, which can be found in the pdfminer
# package in programming.html
def pdf2txt(filename, txtfile):
	import subprocess
	outfile = "-o" + txtfile
	subprocess.call(["pdf2txt.py", outfile, "-c utf-8", filename])
	return

if __name__ == "__main__":
    init()