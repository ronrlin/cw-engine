#!/usr/bin/python
import MySQLdb
import config

# create provision_db
# CREATE DATABASE provision_db;

"""
Instructions on creating environment:

%> mysql -uroot
> CREATE DATABASE provision_db;

> CREATE TABLE provision_lookup(id INT PRIMARY KEY AUTO_INCREMENT, type_name VARCHAR(99), agreement_type VARCHAR(99))

> CREATE TABLE provision_selection(id INT PRIMARY KEY AUTO_INCREMENT, provision_text VARCHAR(99), provision_type INT, score INT)
+----------------+---------+------+-----+---------+----------------+
| Field          | Type    | Null | Key | Default | Extra          |
+----------------+---------+------+-----+---------+----------------+
| id             | int(11) | NO   | PRI | NULL    | auto_increment |
| provision_text | text    | YES  |     | NULL    |                |
| provision_type | int(11) | YES  |     | NULL    |                |
| score          | int(11) | YES  |     | NULL    |                |
+----------------+---------+------+-----+---------+----------------+
> exit
%>

"""

class ProvisionMiner(object):

	def __init__(self):
		# raise on error on failure
		mysql = config.load_mysql()
		self.db = self.connect(host=mysql['host'], user=mysql['user'], passwd=mysql['passwd'], db_name=mysql['db_name'])

	def __del__(self):
		if hasattr(self, 'db'):
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

	ptext = "'Confidential Information' means (whether disclosed directly or indirectly, in writing, electronically, orally, or by inspection or viewing, or in any other form or medium) all proprietary, non-public information of or relating to the Disclosing Party or any of its Affiliates, including but not limited to, financial information, customer lists, supplier lists, business forecasts, software, sales, merchandising and marketing plans and materials, proprietary technology and products, whether or not subject to registration, patent filing or copyright, and all notes, summaries, reports, analyses, compilations, studies and interpretations of any Confidential Information or incorporating any Confidential Information, whether prepared by or on behalf of the Disclosing Party or the Receiving Party.  Confidential Information shall also include the fact that discussions or negotiations are taking place concerning the Transaction between the Disclosing Party and the Receiving Party, and any of the terms, conditions or others facts with respect to any such Transaction, including the status thereof."  
	ptype = "confidential_information"
	pscore = 100
	agreement_type = "nondisclosure"
	miner.create_provision_selection(ptext, ptype, pscore, agreement_type)

	ptext = """The provisions of this Agreement shall not apply to any Confidential Information which:
	(a)	(i) was already known to or in the possession of the Recipient prior to its disclosure pursuant to this Agreement, (ii) was disclosed to the Recipient by a third party not known by Recipient to be under a duty of confidentiality to the Discloser or (iii) which Recipient can establish by competent documentation was independently developed by the Recipient; or
	(b)	is now or hereafter comes into the public domain through no violation of this Agreement by the Recipient; or
	(c) is requested or required by a subpoena or other legal process served upon or otherwise affecting the Recipient.  In such event, the Recipient shall, to the extent permitted by law, notify the Discloser as promptly as is practicable, and the Recipient shall use commercially reasonable efforts to cooperate with the Discloser, at the Discloser’s sole cost and expense, in any lawful effort to contest the validity of such subpoena or legal process.  Notwithstanding the foregoing, the Recipient may, without giving notice to the Discloser, disclose Confidential Information to any governmental agency or regulatory body having or claiming to have authority to regulate or oversee any aspect of the Recipient’s business or the business of the Recipient’s affiliates or representatives; or
	(d) the extent necessary or appropriate to effect or preserve Bank of America’s security (if any) for any obligation due to Bank of America from Company or to enforce any right or remedy or in connection with any claims asserted by or against Bank of America or any of its Representatives or the Company or any other person or entity involved in the Transaction."""
	ptype = "nonconfidential_information"
	pscore = 100
	agreement_type = "nondisclosure"
	miner.create_provision_selection(ptext, ptype, pscore, agreement_type)


	ptext = """MD and COMPANY mutually agree to hold each other’s Proprietary Information in strict confidence, not to disclose such Proprietary Information to any third parties without the written permission of the Disclosing Party, and not to use the other party’s Proprietary Information for its own purposes or for any reason other than for the Purpose.  Other uses are not contemplated and are strictly prohibited; except that, subject to Section 1, Receiving Party may disclose the Disclosing Party’s Proprietary Information only if the Receiving Party is required by law to make any such disclosure that is prohibited or otherwise constrained by this Agreement, provided that the Receiving Party will, to the extent legally permissible, provide the Disclosing Party with prompt written notice of such requirement so that the Disclosing Party may seek, at its own expense, a protective order or other appropriate relief.  Subject to the foregoing sentence, such Receiving Party may furnish only that portion of the Proprietary Information that the Receiving Party is legally compelled or is otherwise legally required to disclose; provided, further, that the Receiving Party shall provide such assistance as the Disclosing Party may reasonably request in obtaining such order or other relief."""

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