#%% load the data from the database
import os
import json
import sqlite3
from pprint import pprint

from dotenv import load_dotenv
load_dotenv()

# get the path of the database
path_db = os.getenv('OUTPUT_DATABASE')
if path_db is None:
    raise Exception('OUTPUT_DATABASE is not set in .env file')

print("* OUTPUT_DATABASE=%s" % path_db)

# create a local database if not exists
conn = sqlite3.connect(path_db)
cursor = conn.cursor()
print('* created a local database if not exists')


#%% load the data from 
rs = cursor.execute('select * from paper_software_names').fetchall()
print('* found %s papers' % len(rs))

ps = []

for r in rs:
    j = json.loads(r[1])
    
    if len(j['software']) > 0:
        ps.append({
            'pmid': r[0],
            'software': j['software']
        })
print('* found %s papers with software names' % len(ps))

pprint(ps)

# dump to a file for further analysis
# json.dump(ps, open('software_names.json', 'w'), indent=2)

# print('* dumped the software names')