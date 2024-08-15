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
# fetchall(): 获取查询的所有结果。
# 将所有行的结果作为一个【列表】返回
# 每一行是一个元组，包含了从数据库中检索的所有列数据。
print('* found %s papers' % len(rs))

ps = []

for r in rs:
    # 存进去的格式是 (pid, json.dumps(software_names))
    # result = {'software': software_names_with_contexts}
    # save_paper_software_names(paper['pmid'], result)
    j = json.loads(r[1])
    if len(j['software']) > 0: # 本paper有stn，才加到结果列表中
        ps.append({
            'pmid': r[0],
            'software': j['software']
        })
print('* found %s papers with software names' % len(ps))

pprint(ps)

# dump to a file for further analysis
# json.dump(ps, open('software_names.json', 'w'), indent=2)

# print('* dumped the software names')