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
corpus = build_corpus()
print("Loading the agreement classifier...")
classifier = get_agreement_classifier_v3(corpus)
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

@app.route('/contract', methods=['GET', 'POST'])
def contract():
	""" Retrieve all contracts for a user. """
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
			contract_data = f.stream.getvalue()

		else:
			d = json.loads(request.data)
			contract_data = d.get('text', None)
			if not contract_data:
				app.logger.error('Could not find field named text in request.data')
				raise InvalidUsage("Did not provide a parameter named 'text'", status_code=400)
			contract_data = d['text']

		# Analyze the contract
		agreement_type = classifier.classify_data(contract_data)
		print("The uploaded agreement was classified as a %s agreement." % agreement_type)
		# Add contract_data to the datastore
		contract_id = datastore.save_contract(contract_data, agreement_type)
		# Return a contract_id 
		return str(contract_id)

@app.route('/contract/<contract_id>', methods=['GET', 'DELETE'])
def handle_contract(contract_id=None):
	""" Retrieve or delete a contract. """
	if request.method == 'GET':
		# Query the database on the contract_id
		contract = datastore.get_contract(contract_id)
		if (contract is None):
			raise InvalidUsage("Contract id %s was not found" % contract_id, status_code=404)

		print(contract)
		# Check the agreement_type
		print("accessing the contract dict...")
		if (contract['agreement_type'] != 'nondisclosure'):
			print("Update the db for now")
			datastore.update_contract(contract_id, 'nondisclosure')
			contract['agreement_type'] = 'nondisclosure'

		agreement_type = contract['agreement_type']
		agreement_text = contract['text']
		print("loading the agreement schema for %s..." % agreement_type)
		schema = AgreementSchema()
		schema.load_schema(agreement_type)

		# Start alignment
		aligner = Alignment(schema=schema, vectorizer=2, all=True)
		paras = aligner.tokenize(agreement_text)
		paras = aligner.simplify(paras)
		aligned_provisions = aligner.align(paras)
		detail = aligner.get_detail(aligned_provisions)

		# Create the JSON response to the browser
		return json.dumps(detail)

	elif request.method == 'DELETE':
		return 'Delete contract ' + contract_id

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