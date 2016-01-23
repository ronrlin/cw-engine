"""`main` is the top level module for your Flask application."""

# Import the Flask Framework
from flask import Flask
from flask import request

# Logging
import logging
from logging.handlers import RotatingFileHandler

# Import json and utilities
import json
from bson.json_util import dumps

# Load ContractWiser modules
from classifier import *
from structure import AgreementSchema
from alignment import Alignment
from helper import WiserDatabase

from werkzeug import secure_filename

UPLOAD_FOLDER = './dump/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'doc', 'docx', 'rtf'])

# Basics
app = Flask(__name__)
handler = RotatingFileHandler('foo.log', maxBytes=10000, backupCount=1)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)
# Note: We don't need to call run() since our application is embedded within
# the App Engine WSGI application server.

print("Loading the datastores...")
datastore = WiserDatabase()
print("Loading the agreement corpus...")
corpus = build_corpus(binary=False)
print("Loading the agreement classifier...")
classifier = get_agreement_classifier_v1(corpus)
print("Application ready to load.")

class InvalidUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        rv['code'] = self.status_code
        return rv

@app.route('/')
def hello():
    """ Return a friendly HTTP greeting. """
    return 'Hello World!'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/contract', methods=['GET', 'POST'])
def contract():
	""" Retrieve all contracts for a user. """
	import requests
	import config
	if request.method == 'GET':
		json_response = {}
		user_record = { 'user' : { 'user_id' : 1 }, 'contracts' : [1, 2, 3, 4] }
		return json.dumps(user_record)

	elif request.method == 'POST':
		# To manually test this service using cURL:
		# curl -X POST -F "data=@filename" http://127.0.0.1:5000/contract		
		# Check that data was POSTed to the service
		contract_data = None
		if not request.data:
			f = request.files['data']
			print(f.filename)

			if f and allowed_file(f.filename):
				import requests
				print("file allowed")
				filename = secure_filename(f.filename)
				print("secure filename is needed: %s" % filename)
				f.save(os.path.join(UPLOAD_FOLDER, filename))
				print("save to filesystem... sucks!")
				with open(os.path.join(UPLOAD_FOLDER, filename),'rb') as file_upload:
					output = file_upload.read()

				tika = config.load_tika()
				tika_url = "http://" + tika['hostname'] + ":" + tika['port'] + "/tika"
				print("Tika is being accessed at %s" % tika_url)
				if (".pdf" in f.filename.lower()):					
					r=requests.put(tika_url, data=output, headers={"Content-type" : "application/pdf", "Accept" : "text/plain"})
					contract_data = r.text.encode("ascii", "replace")
					contract_data = contract_data.replace("?", " ")

				elif (".docx" in f.filename.lower()):
					r=requests.put(tika_url, data=output, headers={"Content-type" : "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "Accept" : "text/plain"})
					contract_data = r.text.encode("ascii", "replace")
					contract_data = contract_data.replace("?", " ")

				elif (".doc" in f.filename.lower()):
					r=requests.put(tika_url, data=output, headers={"Content-type" : "application/msword", "Accept" : "text/plain"})
					contract_data = r.text.encode("ascii", "replace")
					contract_data = contract_data.replace("?", " ")

				elif (".rtf" in f.filename.lower()):
					r=requests.put(tika_url, data=output, headers={"Content-type" : "application/rtf", "Accept" : "text/plain"})
					contract_data = r.text.encode("ascii", "replace")
					contract_data = contract_data.replace("?", " ")

				elif (".txt" in f.filename.lower()): 
					r=requests.put(tika_url, data=output, headers={"Content-type" : "text/plain", "Accept" : "text/plain"})
					contract_data = r.text.encode("ascii", "replace")
					contract_data = contract_data.replace("?", " ")
			else:
				raise InvalidUsage("Did not provide an allowed file format.", status_code=400)

		agreement_type = classifier.classify_data([contract_data])
		print("The uploaded agreement was classified as a %s agreement." % agreement_type)
		# Add contract_data to the datastore
		contract_id = datastore.save_contract(contract_data, agreement_type)
		document = dict()
		document['mainDoc'] = {
			'contract_id' : str(contract_id),
			'agreement_type' : agreement_type,
		}
		return json.dumps(document)

@app.route('/contract/<contract_id>', methods=['GET', 'DELETE'])
def handle_contract(contract_id=None):
	""" Retrieve or delete a contract. """
	if request.method == 'GET':
		print("retrieve a contract")
		# Query the database on the contract_id
		contract = datastore.get_contract(contract_id)
		if (contract is None):
			raise InvalidUsage("Contract id %s was not found" % contract_id, status_code=404)
		schema = AgreementSchema()
		schema.load_schema(contract['agreement_type'])
		# Start alignment
		aligner = Alignment(schema=schema, vectorizer=2, all=True)
		paras = aligner.tokenize(contract['text'])
		paras = aligner.simplify(paras)
		aligned_provisions = aligner.align(paras, version=2)
		detail = aligner.get_detail(aligned_provisions, redline=False)

		with open('log.txt', 'w') as outfile:
			json.dump(detail, outfile, indent=4, sort_keys=True)

		# Create the JSON response to the browser
		return json.dumps(detail)

	elif request.method == 'DELETE':
		return 'Delete contract ' + contract_id

@app.route('/contract/<contract_id>/redline', methods=['GET'])
def handle_redline(contract_id=None):
	contract = datastore.get_contract(contract_id)
	if (contract is None):
		raise InvalidUsage("Contract id %s was not found" % contract_id, status_code=404)
	schema = AgreementSchema()
	schema.load_schema(contract['agreement_type'])
	# Start alignment
	aligner = Alignment(schema=schema, vectorizer=2, all=True)
	paras = aligner.tokenize(contract['text'])
	paras = aligner.simplify(paras)
	aligned_provisions = aligner.align(paras, version=2)
	detail = aligner.get_detail(aligned_provisions, redline=True)
	# Create the JSON response to the browser
	return json.dumps(detail)

@app.route('/contract/<contract_id>/standard', methods=['PUT', 'DELETE'])
def handle_standards(contract_id=None):
	contract = datastore.get_contract(contract_id)
	if (contract is None):
		raise InvalidUsage("Contract id %s was not found" % contract_id, status_code=404)
	schema = AgreementSchema()
	schema.load_schema(contract['agreement_type'])
	if request.method == 'PUT':
		# set the contract as a standard
		datastore.set_standard(contract_id)
		# Start alignment
		aligner = Alignment(schema=schema, vectorizer=2, all=True)
		paras = aligner.tokenize(contract['text'])
		aligned_provisions = aligner.align(paras, version=2)
		# write provisions to provision_db
		for proviso in aligned_provisions:
			ptext = proviso[0]
			ptype = proviso[1]
			pm = ProvisionMiner()
			pm.create_provision_selection(ptext, ptype, 100, 50, contract['agreement_type'], owner_id=contract_id)
		print("contract %s was made a standard." % contract_id)

	elif request.method == 'DELETE':
		# get the contract
		contract = datastore.get_contract(contract_id)
		datastore.unset_standard(contract_id)
		# TODO: delete all the provisions associated to contract_id
		#delete_provision_selection(contract_id)

	# Create the JSON response to the browser
	return True


@app.route('/users')
def users():
	"""Retrieve a list of users."""
	return 'A list of users'

@app.route('/user/<user_id>', methods=['GET', 'PUT'])
def user(user_id=None):
	""" Retrieve or update a user's details. """
	if request.method == 'GET':
		return 'Get user info for ' + user_id

	elif request.method == 'PUT':
		return 'Update user ' + user_id

	elif request.method == 'DELETE':
		return 'Delete user ' + user_id

@app.route('/user', methods=['POST'])
def create_user():
	""" Register a user. """
	return 'Register a user'

@app.errorhandler(404)
def page_not_found(e):
	""" Return a custom 404 error. """
	return 'Sorry, Nothing at this URL.', 404

@app.errorhandler(500)
def application_error(e):
	""" Return a custom 500 error. """
	return 'Sorry, unexpected error: {}'.format(e), 500

@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
	response = json.dumps(error.to_dict())
	return response, error.status_code

if __name__ == '__main__':
    app.run()    