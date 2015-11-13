#!/usr/bin/python
import MySQLdb

# create provision_db
# CREATE DATABASE provision_db;

"""
Instructions on creating environment:

%> mysql -uroot
> CREATE DATABASE provision_db;
> exit
%>

"""

class ProvisionMiner(object):

	def __init__(self):
		# raise on error on failure
		self.db = self.connect()

	def __del__(self):
		self.db.close()

	def connect(self, host="localhost", user="root", passwd="", db_name="provision_db"):
		db = MySQLdb.connect(host=host, user=user, passwd=passwd, db=db_name)
		return db

	def check_provision_lookup(self, ptype, atype):
		""" Check if a provision_lookup record exists for a ptype/atype combination """
		cursor = self.db.cursor(MySQLdb.cursors.DictCursor)
		cursor.execute("SELECT * FROM provision_lookup WHERE type_name=%s AND agreement_type=%s", (ptype, atype))
		pid = 0
		for row in cursor: 
			pid = row["id"]
		cursor.close()
		return pid

	def create_provision_selection(self, ptext, ptype, pscore, agreement_type):
		""" Create a provision selection record. """
		cursor = self.db.cursor(MySQLdb.cursors.DictCursor)
		provision_type_id = check_provision_lookup(ptype, agreement_type)
		if not provision_type_id:
			# create a lookup record if one is lacking
			cursor.execute("INSERT INTO provision_lookup(type_name, agreement_type) VALUES(%s, %s)", (ptype, agreement_type))
			provision_type_id = cursor.lastrowid
		# create the selection record
		cursor.execute("INSERT INTO provision_selection(provision_text, provision_type, score) VALUES(%s, %s, %s)", (ptext, provision_type_id, pscore))
		selection_id = cursor.lastrowid
		self.db.commit()
		cursor.close()
		return selection_id

	def find_better(self, provision_type, agreement_type):
		""" Query to find the best provision """
		query = """select ps.id, ps.provision_text, ps.provision_type, ps.score, pl.agreement_type 
			from provision_selection ps 
			INNER JOIN provision_lookup pl 
			ON pl.id=ps.provision_type
			WHERE pl.type_name = %s
			AND pl.agreement_type = %s
			ORDER BY ps.score DESC  
			"""
		cursor = self.db.cursor(MySQLdb.cursors.DictCursor)
		cursor.execute(query, (provision_type, agreement_type))
		result = cursor.fetchone()
		if not result:
			print("throw an exception")
			result = dict()
			result['provision_text'] = "There will be new text here."			
		cursor.close()
		return result['provision_text'] 

def find_better(provision_type, agreement_type, score=50):
	db = MySQLdb.connect(host="localhost", user="root", passwd="", db="provision_db")
	cursor = db.cursor(MySQLdb.cursors.DictCursor)
	query = """select ps.id, ps.provision_text, ps.provision_type, ps.score, pl.agreement_type 
		from provision_selection ps 
		INNER JOIN provision_lookup pl 
		ON pl.id=ps.provision_type
		WHERE pl.type_name = %s
		AND pl.agreement_type = %s
		"""
	# consider ordering by score or something
	cursor.execute(query, (provision_type, agreement_type))
	result = cursor.fetchone()
	return result['provision_text']

def sample_db_query():
	db = MySQLdb.connect(host="localhost", user="root", passwd="", db="provision_db")
	cursor = db.cursor(MySQLdb.cursors.DictCursor)
	cursor.execute("SELECT * FROM users")
	rows = cursor.fetchall()
	print(cursor.description)
	return

def setup_new():
	miner = ProvisionMiner()

	ptext = """ Exclusions from Confidential Information. Receiving Party's obligations under this Agreement do not extend to information that is: (a) publicly known at the time of disclosure or subsequently becomes publicly known through no fault of the Receiving Party; (b) discovered or created by the Receiving Party before disclosure by Disclosing Party; (c) learned by the Receiving Party through legitimate means other than from the Disclosing Party or Disclosing Party's representatives; (d) is disclosed by Receiving Party with Disclosing Party's prior written approval; (e) is requested or required by a subpoena or other legal process served upon or otherwise affecting the Recipient.  In such event, the Recipient shall, to the extent permitted by law, notify the Discloser as promptly as is practicable, and the Recipient shall use commercially reasonable efforts to cooperate with the Discloser, at the Discloser's sole cost and expense, in any lawful effort to contest the validity of such subpoena or legal process.  Notwithstanding the foregoing, the Recipient may, without giving notice to the Discloser, disclose Confidential Information to any governmental agency or regulatory body having or claiming to have authority to regulate or oversee any aspect of the Recipient's business or the business of the Recipient's affiliates or representatives. """
	ptype = "confidential_information"
	pscore = 100
	agreement_type = "nondisclosure"
	miner.create_provision_selection(ptext, ptype, pscore, agreement_type)

	print("delete")
	del(miner)

def setup():
	db = MySQLdb.connect(host="localhost", user="root", passwd="", db="provision_db")
	cursor = db.cursor()

	ptext = """ Exclusions from Confidential Information. Receiving Party's obligations under this Agreement do not extend to information that is: (a) publicly known at the time of disclosure or subsequently becomes publicly known through no fault of the Receiving Party; (b) discovered or created by the Receiving Party before disclosure by Disclosing Party; (c) learned by the Receiving Party through legitimate means other than from the Disclosing Party or Disclosing Party's representatives; (d) is disclosed by Receiving Party with Disclosing Party's prior written approval; (e) is requested or required by a subpoena or other legal process served upon or otherwise affecting the Recipient.  In such event, the Recipient shall, to the extent permitted by law, notify the Discloser as promptly as is practicable, and the Recipient shall use commercially reasonable efforts to cooperate with the Discloser, at the Discloser's sole cost and expense, in any lawful effort to contest the validity of such subpoena or legal process.  Notwithstanding the foregoing, the Recipient may, without giving notice to the Discloser, disclose Confidential Information to any governmental agency or regulatory body having or claiming to have authority to regulate or oversee any aspect of the Recipient's business or the business of the Recipient's affiliates or representatives. """
	ptype = "nonconfidential_information"
	pscore = 100
	agreement_type = "nondisclosure"
	pid = create_provision_selection(ptext, ptype, pscore, agreement_type)	
	print("%d was created in provision_selection table" % pid)

	# create provision_selection
	cursor.execute("DROP TABLE IF EXISTS provision_selection")
	cursor.execute("CREATE TABLE provision_selection(id INT PRIMARY KEY AUTO_INCREMENT, provision_text TEXT, provision_type INT, score INT)")
	cursor.execute("INSERT INTO provision_selection(provision_text, provision_type, score) VALUES('Text will be here.', 1, 50)")
	# create provision_lookup
	cursor.execute("DROP TABLE IF EXISTS provision_lookup")
	cursor.execute("CREATE TABLE provision_lookup(id INT PRIMARY KEY AUTO_INCREMENT, type_name VARCHAR(99), agreement_type VARCHAR(99))")
	cursor.execute("INSERT INTO provision_lookup(type_name, agreement_type) VALUES('confidential_information', 'nondisclosure')")
	cursor.execute("INSERT INTO provision_lookup(type_name, agreement_type) VALUES('confidential_information', 'another_type')")
	db.commit()
	cursor.close()
	db.close()
	print("end of setup.")

def check_provision_lookup(ptype, atype):
	""" Check if a provision_lookup record exists for a ptype/atype combination """
	db = MySQLdb.connect(host="localhost", user="root", passwd="", db="provision_db")
	cursor = db.cursor(MySQLdb.cursors.DictCursor)
	cursor.execute("SELECT * FROM provision_lookup WHERE type_name=%s AND agreement_type=%s", (ptype, atype))
	pid = 0
	for row in cursor: 
		pid = row["id"]
	cursor.close()
	db.close()
	return pid

def create_provision_selection(ptext, ptype, pscore, agreement_type):
	""" Create a provision selection record. """
	db = MySQLdb.connect(host="localhost", user="root", passwd="", db="provision_db")
	cursor = db.cursor(MySQLdb.cursors.DictCursor)
	provision_type_id = check_provision_lookup(ptype, agreement_type)
	if not provision_type_id:
		# create a lookup record if one is lacking
		cursor.execute("INSERT INTO provision_lookup(type_name, agreement_type) VALUES(%s, %s)", (ptype, agreement_type))
		provision_type_id = cursor.lastrowid
	# create the selection record
	cursor.execute("INSERT INTO provision_selection(provision_text, provision_type, score) VALUES(%s, %s, %s)", (ptext, provision_type_id, pscore))
	selection_id = cursor.lastrowid
	db.commit()
	cursor.close()
	db.close()
	return selection_id


def test2():
	ptext = """ Confidential Information does not include information:  
(i) generally known on a non-confidential basis to the public (through no fault of the Receiving Party); 
(ii) within the Receiving Party's possession prior to the receipt of the information from the Disclosing Party, or otherwise lawfully obtained by the Receiving Party, without there being a violation of a restriction on disclosure; 
(iii) independently developed by the Receiving Party without use of information provided by the Disclosing Party; or 
(iv) disclosed by the Receiving Party with the Disclosing Party's prior written approval.
	"""
	ptype = "nonconfidential_information"
	pscore = 51
	agreement_type = "convertible_debt"
	pid = create_provision_selection(ptext, ptype, pscore, agreement_type)	
	print("%d was created in provision_selection table" % pid)

	ptext = """ Exclusions from Confidential Information. Receiving Party's obligations under this Agreement do not extend to information that is: (a) publicly known at the time of disclosure or subsequently becomes publicly known through no fault of the Receiving Party; (b) discovered or created by the Receiving Party before disclosure by Disclosing Party; (c) learned by the Receiving Party through legitimate means other than from the Disclosing Party or Disclosing Party's representatives; (d) is disclosed by Receiving Party with Disclosing Party's prior written approval; (e) is requested or required by a subpoena or other legal process served upon or otherwise affecting the Recipient.  In such event, the Recipient shall, to the extent permitted by law, notify the Discloser as promptly as is practicable, and the Recipient shall use commercially reasonable efforts to cooperate with the Discloser, at the Discloser's sole cost and expense, in any lawful effort to contest the validity of such subpoena or legal process.  Notwithstanding the foregoing, the Recipient may, without giving notice to the Discloser, disclose Confidential Information to any governmental agency or regulatory body having or claiming to have authority to regulate or oversee any aspect of the Recipient's business or the business of the Recipient's affiliates or representatives. """
	ptype = "nonconfidential_information"
	pscore = 100
	agreement_type = "convertible_debt"
	pid = create_provision_selection(ptext, ptype, pscore, agreement_type)	
	print("%d was created in provision_selection table" % pid)

#MARKED FOR REMOVAL
def test():
	pm = ProvisionMiner()
	provision_type = "confidential_information"
	agreement_type = "nondisclosure"
	alt_text = pm.find_better(provision_type, agreement_type)
	print(alt_text)

if __name__ == "__main__":
	""" bypass main() function """
	pass	