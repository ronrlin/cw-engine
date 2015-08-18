"""`main` is the top level module for your Flask application."""

# Import the Flask Framework
from flask import Flask
from flask import request

app = Flask(__name__)
# Note: We don't need to call run() since our application is embedded within
# the App Engine WSGI application server.

@app.route('/')
def hello():
    """Return a friendly HTTP greeting."""
    return 'Hello World!'

@app.route('/contract', methods=['GET', 'POST'])
def contract():
	"""Retrieve all contracts for a user."""
	if request.method == 'GET':
		return 'List of contracts for this user'
	elif request.method == 'POST':
		return	'Upload and process a new contract '
	else:
		return 'An error has occured'

@app.route('/contract/<contract_id>', methods=['GET', 'DELETE'])
def handle_contract(contract_id=None):
	""" Retrieve or delete a contract. """
	if request.method == 'GET':
		return 'Retrieve contract ' + contract_id
	elif request.method == 'DELETE':
		return 'Delete contract ' + contract_id
	else:  
		return 'An error has occured'

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
	else:  
		return 'An error has occured'

@app.route('/user', methods=['POST'])
def create_user():
	""" Register a user. """
	return 'Register a user'

@app.errorhandler(404)
def page_not_found(e):
	"""Return a custom 404 error."""
	return 'Sorry, Nothing at this URL.', 404

@app.errorhandler(500)
def application_error(e):
	"""Return a custom 500 error."""
	return 'Sorry, unexpected error: {}'.format(e), 500
