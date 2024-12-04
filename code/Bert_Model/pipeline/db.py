import os
import json
import sqlite3
from dotenv import load_dotenv
load_dotenv()


# get the path of the database
def create_db(start_year, end_year, sample_size, shots_num):
    global path_db
    path_db = os.getenv('OUTPUT_DATABASE', None)
    path_db = path_db + f"_{start_year}_{end_year}_{sample_size}_{shots_num}" + ".db"
    if path_db is None:
        raise Exception('OUTPUT_DATABASE is not set in .env file')

    print("* OUTPUT_DATABASE=%s" % path_db)
    # create a local database if not exists
    global conn
    global c
    conn = sqlite3.connect(path_db) 
    # 和数据库的连接对象，是与SQLite数据库交互的桥梁，它管理着整个数据库连接的生命周期

    c = conn.cursor() # 创建游标
    # c.execute(...)：执行一个 SQL 命令，下面建立的db叫paper_software_namse

    c.execute(''' 
    CREATE TABLE IF NOT EXISTS paper_software_names (
        pid TEXT PRIMARY KEY,
        pubdate TEXT,
        journal TEXT,
        mesh_terms TEXT,
        authors TEXT,
        software_names TEXT
    )
    ''')
    # pid是TEXT类型，而且是主键（唯一性）primary key

    conn.commit() # 提交当前事务，将所有的更改保存到数据库文件中。
    print('* created a local database if not exists')




# --- 把pid的software_names放入 --- # 
def save_paper_software_names(pid, pubdate, journal, mesh_terms, authors, software_names):
    '''
    Save the extracted software names for a paper
    '''
    c.execute('INSERT INTO paper_software_names VALUES (?, ?, ?, ?, ?, ?)', (pid, pubdate, journal,\
                                                                     mesh_terms, authors, json.dumps(software_names)))
    conn.commit()
    return software_names


# --- 只提取一个软件名（如有），只是用来看之前是否提取过，是不是空的啦 --- #
def load_paper_software_names(pid):
    '''
    Load the extracted software names for a paper
    '''
    c.execute('SELECT software_names FROM paper_software_names WHERE pid=?', (pid,))
    software_names = c.fetchone()
    # c.fetchone() 从查询结果中获取【第一行数据（如果存在）】。
    # fetchone() 方法返回一个包含结果的元组。
    # 如果查询没有返回结果，fetchone() 会返回 None。
    if software_names is not None:
        return json.loads(software_names[0])
    else:
        return None


# --- 删掉pid对应的那条数据 --- # 
def delete_paper_software_names(pid):
    '''
    Delete the extracted software names for a paper
    '''
    c.execute('DELETE FROM paper_software_names WHERE pid=?', (pid,))
    conn.commit()
    return pid



if __name__ == "__main__":
    pass