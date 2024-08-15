import os
import json
import sqlite3

from dotenv import load_dotenv
load_dotenv()

# get the path of the database
path_db = os.getenv('OUTPUT_DATABASE', None)
if path_db is None:
    raise Exception('OUTPUT_DATABASE is not set in .env file')

print("* OUTPUT_DATABASE=%s" % path_db)

# create a local database if not exists
conn = sqlite3.connect(path_db)
c = conn.cursor()
c.execute('''
CREATE TABLE IF NOT EXISTS paper_software_names (
    pid TEXT PRIMARY KEY,
    software TEXT
)
''')
conn.commit()
print('* created a local database if not exists')



def save_paper_software_names(pid, software_names):
    '''
    Save the extracted software names for a paper
    '''
    c.execute('INSERT INTO paper_software_names VALUES (?, ?)', (pid, json.dumps(software_names)))
    conn.commit()
    return software_names


def load_paper_software_names(pid):
    '''
    Load the extracted software names for a paper
    '''
    c.execute('SELECT software FROM paper_software_names WHERE pid=?', (pid,))
    software_names = c.fetchone()
    if software_names is not None:
        return json.loads(software_names[0])
    else:
        return None

def delete_paper_software_names(pid):
    '''
    Delete the extracted software names for a paper
    '''
    c.execute('DELETE FROM paper_software_names WHERE pid=?', (pid,))
    conn.commit()
    return pid