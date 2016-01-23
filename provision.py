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

| owner_id		 | int(11) | YES  |     | NULL    |                |
| score          | int(11) | YES  |     | NULL    |                |

scope varchar(99)
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

	def create_provision_selection(self, ptext, ptype, pscore, comscore, agreement_type, owner_id=None):
		""" Create a provision selection record. """
		cursor = self.db.cursor(MySQLdb.cursors.DictCursor)
		provision_type_id = self.check_provision_lookup(ptype, agreement_type)
		if not provision_type_id:
			# create a lookup record if one is lacking
			cursor.execute("INSERT INTO provision_lookup(type_name, agreement_type) VALUES(%s, %s)", (ptype, agreement_type))
			provision_type_id = cursor.lastrowid
		# create the selection record
		cursor.execute("INSERT INTO provision_selection(provision_text, provision_type, score, complexity_score, owner_id) VALUES(%s, %s, %s, %s, %s)", (ptext, provision_type_id, pscore, comscore, owner_id))
		selection_id = cursor.lastrowid
		self.db.commit()
		cursor.close()
		return selection_id

	def delete_provision_selection(self, owner_id):
		cursor = self.db.cursor(MySQLdb.cursors.DictCursor)
		cursor.execute("DELETE FROM provision_selection WHERE owner_id = %s", (owner_id))
		self.db.commit()
		cursor.close()
		return

	def find_better(self, provision_type, agreement_type):
		""" Query to find the best provision """
		query = """select ps.id, ps.provision_text, ps.provision_type, ps.score, ps.complexity_score, pl.agreement_type, ps.owner_id 
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
			result['provision_text'] = ""
		cursor.close()
		return result['provision_text'] 

def create_system():
	print("Create the provision_db.")
	create_provision_dbs()
	init_data()
	print("Ready to be smart.")

def get_description(provision_name):

	if provision_name == "confidential_information":
		return "The Confidential Information clause defines what the parties deem to be proprietary and protected by the terms of the agreement. The clause typically defines what is confidential."
	if provision_name == "nonconfidential_information":
		return "This provision specifies what might be explicitly excluded from from being considered protected information."
	elif provision_name == "no_future_obligation":
		return "This provision states that neither party has any obligation under this agreement to purchase from or provide to the other party any products or services, or to enter into any other agreement."
	elif provision_name == "ownership_of_information":
		return "This provision usually specifies that all proprietary rights are retained by the disclosing party and that the disclosure of information does not convey any right or license to use the information other than for the stated purpose. In some cases, an agreement might be specific about what information or classes of information would fall under these rights."
	elif provision_name == "obligation_of_receiving_party":
		return "This provision is telling you that the receiving party is promising to do something in connection with executing this agreement."
	elif provision_name == "breach":
		return "This provision specifies the terms or conditions under which it would be understood that the agreement is being violated.  In some cases, parties might also specify what damages or claims would be made in the event of a breach. "
	elif provision_name == "indemnification":
		return "The receiving party shall indemnify the disclosing party against all claims, liability, costs, and expenses (including attorney fees) arising from any third party claim or proceeding brought against the disclosing party that relates to or arises out of any disclosure of Confidential Information by the receiving party or its Authorized Representatives in breach of this agreement."
	elif provision_name == "time_period":
		return "This provision defines the period of time during which the parties must keep information confidential. The clause may provide for a fixed or undefined term; others may impose obligations of confidential until such time as the information becomes public. The clause must be read in conjunction with the termination provision."
	elif provision_name == "representations_and_warranties":
		return "This clause states that the parties make no guarantees to the accuracy or completeness of any confidential information."
	elif provision_name == "severability":
		return "The Severability clause states that the terms of the contract are independent of each another, so that if a term in the contract is deemed unenforceable by a court, the contract as a whole will not be deemed unenforceable. The severability clause is frequently one of the most variable (or inconsistent) clauses in an agreement as lawyers creatively attempt to maximize the scope of restrictive covenants."
	elif provision_name == "waiver":
		return "The Waiver (or No Waiver) clause ensures that a party's failure to enforce its contractual rights, whether intentionally or not intentionally, does not result in the party waiving future rights or remedies for their breach. A party may decide not to strictly enforce the provisions of the contract on a particular occasion or under certain circumstances. It might, for example, accept late payments without insisting on an agreed-upon penalty. "
	elif provision_name == "integration":
		return "This clause declares a contract to be the complete and final agreement between the parties. A contract that has such a clause may be deemed an integrated contract, and any previous negotiations in which the parties to the contract had considered different terms will be deemed superseded by the final writing. The existence of such a term is normally not conclusive proof that no varied or additional conditions exist with respect to the performance of the contract beyond those that are in the writing but instead is simply evidence of that fact."
	elif provision_name == "relationship":
		return "The Relationship of the Parties or No Partnership clause says that the parties do not have any partnership or agency relationship. This means that the agreement does not imply that there is any partnership or relationship between the parties, and the only rights or obligations are explicitly specified in the contract above."
	elif provision_name == "governing_law_jurisdiction":
		return "This clause specifies the location where the laws of a mutually agreed upon jurisdiction will govern the interpretation and enforcement of the terms of the contract."
	elif provision_name == "preamble":
		return "The preamble is an introductory statement of facts or assumptions upon which a contract is based.  It will typically contain names, addresses and should specify the date an agreement goes into effect."
	elif provision_name == "recital":
		return "A recital sets out the background to the contract. It usually asserts some basic facts and definitions and then precedes the legally binding promises that your contract will make."
	else:
		return "No description available at this time."

def init_data():

	print("ERROR! Check provision.py and consider revisions to ProvisionStatistics!!!")
	return

	from statistics import ProvisionStatistics
	miner = ProvisionMiner()

	############ confidential_information ################
	ptype = "confidential_information"
	pstats = ProvisionStatistics(ptype)

	ptext = """ Exclusions from Confidential Information. Receiving Party's obligations under this Agreement do not extend to information that is: (a) publicly known at the time of disclosure or subsequently becomes publicly known through no fault of the Receiving Party; (b) discovered or created by the Receiving Party before disclosure by Disclosing Party; (c) learned by the Receiving Party through legitimate means other than from the Disclosing Party or Disclosing Party's representatives; (d) is disclosed by Receiving Party with Disclosing Party's prior written approval; (e) is requested or required by a subpoena or other legal process served upon or otherwise affecting the Recipient.  In such event, the Recipient shall, to the extent permitted by law, notify the Discloser as promptly as is practicable, and the Recipient shall use commercially reasonable efforts to cooperate with the Discloser, at the Discloser's sole cost and expense, in any lawful effort to contest the validity of such subpoena or legal process.  Notwithstanding the foregoing, the Recipient may, without giving notice to the Discloser, disclose Confidential Information to any governmental agency or regulatory body having or claiming to have authority to regulate or oversee any aspect of the Recipient's business or the business of the Recipient's affiliates or representatives. """
	comscore = pstats.calculate_complexity(ptext)
	pscore = 100
	agreement_type = "nondisclosure"
	miner.create_provision_selection(ptext, ptype, pscore, comscore, agreement_type)

	ptext = "'Confidential Information' means (whether disclosed directly or indirectly, in writing, electronically, orally, or by inspection or viewing, or in any other form or medium) all proprietary, non-public information of or relating to the Disclosing Party or any of its Affiliates, including but not limited to, financial information, customer lists, supplier lists, business forecasts, software, sales, merchandising and marketing plans and materials, proprietary technology and products, whether or not subject to registration, patent filing or copyright, and all notes, summaries, reports, analyses, compilations, studies and interpretations of any Confidential Information or incorporating any Confidential Information, whether prepared by or on behalf of the Disclosing Party or the Receiving Party.  Confidential Information shall also include the fact that discussions or negotiations are taking place concerning the Transaction between the Disclosing Party and the Receiving Party, and any of the terms, conditions or others facts with respect to any such Transaction, including the status thereof."  
	comscore = pstats.calculate_complexity(ptext)
	pscore = 100
	agreement_type = "nondisclosure"
	miner.create_provision_selection(ptext, ptype, pscore, comscore, agreement_type)

	############ nonconfidential_information ################
	ptype = "nonconfidential_information"
	pstats = ProvisionStatistics(ptype)

	ptext = """The provisions of this Agreement shall not apply to any Confidential Information which:
	(a)	(i) was already known to or in the possession of the Recipient prior to its disclosure pursuant to this Agreement, (ii) was disclosed to the Recipient by a third party not known by Recipient to be under a duty of confidentiality to the Discloser or (iii) which Recipient can establish by competent documentation was independently developed by the Recipient; or
	(b)	is now or hereafter comes into the public domain through no violation of this Agreement by the Recipient; or
	(c) is requested or required by a subpoena or other legal process served upon or otherwise affecting the Recipient.  In such event, the Recipient shall, to the extent permitted by law, notify the Discloser as promptly as is practicable, and the Recipient shall use commercially reasonable efforts to cooperate with the Discloser, at the Discloser's sole cost and expense, in any lawful effort to contest the validity of such subpoena or legal process.  Notwithstanding the foregoing, the Recipient may, without giving notice to the Discloser, disclose Confidential Information to any governmental agency or regulatory body having or claiming to have authority to regulate or oversee any aspect of the Recipient's business or the business of the Recipient's affiliates or representatives; or
	(d) the extent necessary or appropriate to effect or preserve Bank of America's security (if any) for any obligation due to Bank of America from Company or to enforce any right or remedy or in connection with any claims asserted by or against Bank of America or any of its Representatives or the Company or any other person or entity involved in the Transaction."""
	comscore = pstats.calculate_complexity(ptext)
	pscore = 100
	agreement_type = "nondisclosure"
	miner.create_provision_selection(ptext, ptype, pscore, comscore, agreement_type)

	############ obligation_of_receiving_party ################
	ptype = "obligation_of_receiving_party"
	pstats = ProvisionStatistics(ptype)

	ptext = """COUNTERPARTY and COMPANY mutually agree to hold each other's Proprietary Information in strict confidence, not to disclose such Proprietary Information to any third parties without the written permission of the Disclosing Party, and not to use the other party's Proprietary Information for its own purposes or for any reason other than for the Purpose.  Other uses are not contemplated and are strictly prohibited; except that, subject to Section 1, Receiving Party may disclose the Disclosing Party's Proprietary Information only if the Receiving Party is required by law to make any such disclosure that is prohibited or otherwise constrained by this Agreement, provided that the Receiving Party will, to the extent legally permissible, provide the Disclosing Party with prompt written notice of such requirement so that the Disclosing Party may seek, at its own expense, a protective order or other appropriate relief.  Subject to the foregoing sentence, such Receiving Party may furnish only that portion of the Proprietary Information that the Receiving Party is legally compelled or is otherwise legally required to disclose; provided, further, that the Receiving Party shall provide such assistance as the Disclosing Party may reasonably request in obtaining such order or other relief."""
	comscore = pstats.calculate_complexity(ptext)
	pscore = 100
	agreement_type = "nondisclosure"
	miner.create_provision_selection(ptext, ptype, pscore, comscore, agreement_type)

	############ waiver ################
	ptype = "waiver"
	pstats = ProvisionStatistics(ptype)

	ptext = """No waiver of any provisions or of any right or remedy hereunder shall be effective unless in writing and signed by both party's authorized representatives.  No delay in exercising, or no partial exercise of any right or remedy hereunder, shall constitute a waiver of any right or remedy, or future exercise thereof, nor shall any such delay or partial exercise change the character of the Confidential Information as such."""
	comscore = pstats.calculate_complexity(ptext)
	pscore = 100
	agreement_type = "nondisclosure"
	miner.create_provision_selection(ptext, ptype, pscore, comscore, agreement_type)

	############ severability ################
	ptype = "severability"
	pstats = ProvisionStatistics(ptype)

	ptext = """If any provision of this Agreement is held to be illegal, invalid or unenforceable under present or future laws effective during the term hereof, such provisions shall be fully severable; this Agreement shall be construed and enforced as if such severed provisions had never comprised a part hereof; and the remaining provisions of this Agreement shall remain in full force and effect and shall not be affected by the severed provision or by its severance from this Agreement."""
	comscore = pstats.calculate_complexity(ptext)
	pscore = 100
	agreement_type = "nondisclosure"
	miner.create_provision_selection(ptext, ptype, pscore, comscore, agreement_type)

	############ integration ################
	ptype = "integration"
	pstats = ProvisionStatistics(ptype)

	ptext = """This Agreement supersedes in full all prior discussions and agreements between the Parties relating to the Confidential Information, constitutes the entire agreement between the Parties relating to the Confidential Information, and may be amended, modified or supplemented only by a written document signed by an authorized representative of each Party."""
	comscore = pstats.calculate_complexity(ptext)
	pscore = 100
	agreement_type = "nondisclosure"
	miner.create_provision_selection(ptext, ptype, pscore, comscore, agreement_type)

	print("Clean up the miner.")
	del(miner)

def create_provision_dbs():
	""" """
	# get connection info from config.py
	mysql = config.load_mysql()
	db = MySQLdb.connect(host=mysql['host'], user=mysql['user'], passwd=mysql['passwd'], db=mysql['db_name'])
	#db = MySQLdb.connect(host="localhost", user="root", passwd="", db="provision_db")
	cursor = db.cursor()
	# create provision_selection
	cursor.execute("DROP TABLE IF EXISTS provision_selection")
	cursor.execute("CREATE TABLE provision_selection(id INT PRIMARY KEY AUTO_INCREMENT, provision_text TEXT, provision_type INT, score INT, complexity_score INT, owner_id VARCHAR(99))")
	# create provision_lookup
	cursor.execute("DROP TABLE IF EXISTS provision_lookup")
	cursor.execute("CREATE TABLE provision_lookup(id INT PRIMARY KEY AUTO_INCREMENT, type_name VARCHAR(99), agreement_type VARCHAR(99))")
	db.commit()
	cursor.close()
	db.close()

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