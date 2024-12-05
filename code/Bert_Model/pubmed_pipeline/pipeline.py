
# TODO 1: 创建db
# TODO 2：调用Bert的inference代码
# TODO 3：后处理，上传huggingface

#%% run demo
# import openai
import json
import time
import os
import sys
import sqlite3
import db as db
from db import *
import duckdb
# DuckDB 是一个嵌入式数据库，类似于 SQLite，但更专注于数据分析工作负载。它能够高效地处理数据，并且支持 SQL 查询。
import pysbd
from tqdm import tqdm
import pandas as pd
from openai import OpenAI
from pprint import pprint

# 名字来源于 “pretty-print”。
# 它用于以一种更加可读的方式格式化和输出数据结构，尤其是复杂的嵌套数据结构（如字典、列表、元组等）。
# --- 复杂输出结构时可用！

from dotenv import load_dotenv
from datasets import Dataset, DatasetDict



# ----------------------- 加载.env文件中的信息 ------------------- #
load_dotenv()
print('* loaded configs')
print('* OPENAI_API_BASE_URL=%s' % os.getenv('OPENAI_API_BASE_URL'))
print('* OPENAI_API_MODEL=%s' % os.getenv('OPENAI_API_MODEL'))
print('* PUBMED_DATA_TSV=%s' % os.getenv('PUBMED_DATA_TSV'))
print('* OUTPUT_DATABASE=%s' % os.getenv('OUTPUT_DATABASE'))
print('* HF_TOKEN=%s' % os.getenv('HF_TOKEN'))
print('* PYTHONPATH=%s' % os.getenv('PYTHONPATH'))
EMAIL_FOLDER = os.getenv('EMAIL_FOLDER')
sys.path.append(EMAIL_FOLDER)
from Email_senting import send_email
print('* loaded all libraries')
from Bert_Model.model_code.ner import *



# --- 调用model，进行是BIO tag的inference_file进行处理。返回一个txt文件 --- #
def extract(inference_file):
    
    '''
    Extract something from the abstract of a paper based on the given prompt template.
    '''

    try:
        main(args, inference_file, pipeline=True)
    except Exception as e:
        print(f'! error: {e}')
        # print full stack
        import traceback
        traceback.print_exc()
        return None

# create a segmenter for sentence splitting
# 创建了一个【句子分割器（segmenter）】，用于将文本分割成独立的句子。
# 使用了 pysbd 库，专门用于[句子边界检测]的库，特别适合处理多语言的复杂文本。
# 通过识别常见的句子边界符号（如句号、问号、感叹号等）和处理语言特有的规则来准确地将文本分割成句子
# clean 参数决定了分割器是否在分割前对文本进行清理。
# - True，分割器会移除一些不必要的空白符或修复常见的格式问题。
# - False，分割器会按照原始文本进行分割。

sent_segmenter = pysbd.Segmenter(language="en", clean=False)


# --- paper中把抽取出来的entity的所在句子给拿出来，为了后续做多模型的评判，看抽取出来的是否准确 --- # 
def get_contexts(entities, paper):
    '''
    Get context for the identified entities.
    For example, the input entities may be: 
    ["MetaMap", "word2vec"],
    and the output entities should be:
    [
        {"name": "MetaMap", "context": "MetaMap is a valuable tool for processing biomedical texts to identify concepts."},
        {"name": "word2vec", "context": "For unsupervised training, the phrase and word2vec models used abstracts related to clinical decision support as input."}
    ]
    Basically, the context should be the text snippet from the abstract that contains the entity.
    '''

    ents = []

    # process the abstract first
    abstract = paper['abstract']
    sents = sent_segmenter.segment(abstract) # 分割器分割摘要
    for entity in entities:
        # search this entity in all sentences
        # this entity may appear in multiple sentences
        contexts = []
        
        for sent in sents:

            # the entity should be a substring of the sentence
            # but sometimes the entity may be in different forms
            # e.g., "MetaMap" vs. "metamap"
            # so we use 【lower case for comparison】
            no_spaces_sent = sent.lower().replace(" ", "")
            no_spaces_entity = entity.lower().replace(" ","")
            if no_spaces_entity in no_spaces_sent: # 字符串用in就可以！不用find来找啦
                contexts.append(sent)
        if len(contexts) != 0:
            ents.append({
                'name': entity,
                'contexts': contexts
            })
                
    return ents



# --- 依据给定的预处理BIO文件，Bert做抽取 --- #

def extract_and_save_to_db(inference_file):
    '''
    Extract software names from the given paper and save the result to the database.
    The paper should be a dictionary with the following keys
    - pmid: the PMID of the paper
    - title: the title of the paper
    - abstract: the abstract of the paper
    '''

    # not found, extract the software names
    # extract using the openai model
    extract(inference_file) # 生成一个抽取的BIO文件

    exit()
    # --- 后处理 --- # 
    # --- TODO: 以下是对单文件进行的处理，需要修改成对一群文件 --- #
    if tmp is None:
        # no software names found or error
        result = {'software': []}
    else:
        # add context to the extracted entities
        try:
            software_names_with_contexts = get_contexts(tmp['software'], paper)
        except Exception as e:
            # 可能model返回的格式不是字典 or 是字典，但是没有“software”这个键，导致上一句执行错误
            # 【对model不按指示推理设置保险 -- 不会在没"software"这个键上报错】
            print(f'! error: {e}')
            print(f'! failed to get context for {tmp}')
            software_names_with_contexts = []

        result = {'software': software_names_with_contexts}
    # --- process authors' name --- #
    paper["authors"] = paper["authors"].replace("|", " ")
    db.save_paper_software_names(paper['pmid'], paper["pubdate"], \
                                 paper["journal"], paper["mesh_terms"], \
                                 paper["authors"], result)
    #(pid, pubdate, journal,\
    # mesh_terms, authors, json.dumps(software_names)))

    current = dict()
    current["pmid"] = paper['pmid']
    current["pubdate"] = paper['pubdate']
    current["software_name"] = result["software"]
    current["journal"] = paper["journal"]
    current["mesh terms"] = paper["mesh_terms"]
    current["authors"] = paper["authors"]
    return current



# --- 抽取结果上传huggingface --- # 
def upload_hugging_face(data, start_year, end_year, each_year_sample_size):

    HF_TOKEN=os.getenv("HF_TOKEN")
    print("* Huggingface token", HF_TOKEN)
    # data是一个列表形式，每一个元素都是字典
    data = Dataset.from_dict({key: [d[key] for d in data] for key in data[0].keys()})
    dataset_dict = DatasetDict({
        'train': data,
        'valid': data,
        'test': data
    })

    # 上传数据集到 Hugging Face
    # 设置Hugging Face API Token，确保已登录Hugging Face并生成API密钥
    if not HF_TOKEN:
        raise ValueError("Please set your Hugging Face API token as an environment variable named 'HF_TOKEN'.")

    dataset_dict.push_to_hub(f'JerrySiRi/SWN_Bert_based_{start_year}_{end_year}_{each_year_sample_size}', token=HF_TOKEN)


# --- 总抽取函数 --- # 
def extract_and_save_samples(
    output_file="../../../datasets/PubMed",
    sample_size=10, 
    pmid_filter='1=1',
    start_year = 2010,
    end_year = 2023,
    main_run = True,
    percent = 1
):
    '''
    Extract software names from all papers in the database.
    '''
    # print(f'* extracting software names of {sample_size} papers where {pmid_filter}')
    #%% load the data from 
    duck_conn = duckdb.connect()
    # 这行代码调用了 duckdb 模块的 connect 方法，创建了一个【与DuckDB 数据库的连接】。
    # 此连接对象允许你执行 SQL 查询、操作数据库表、插入或检索数据等。

    # load data
    path_data = os.getenv('PUBMED_DATA_TSV')

    print('* loading data from %s (whole pubmed data is about 60G, it may take a few minutes)' % path_data)
    # 原来PubMed的csv文件，转化成duckdb数据库
    duck_conn.execute(f"""
        CREATE TABLE IF NOT EXISTS papers AS
        SELECT *
        FROM read_csv_auto('{path_data}', delim='\\t', header=True)
    """)
    print('* loaded data from %s' % path_data)

    # ---------- 筛选逻辑，筛选生成dataframe --------- # 
    # sample_size = 每年的LIMI
    # 按照pubdate设置分区，再在分区中显示sample_size
    if main_run == True:
        df = duck_conn.execute(f"""
    WITH year_partition AS (
        SELECT *, 
            ROW_NUMBER() OVER (PARTITION BY pubdate 
                                    ORDER BY RANDOM()) AS rn
        FROM papers
        WHERE abstract IS NOT NULL 
            AND abstract <> ''
            AND {pmid_filter}
            AND pubdate BETWEEN {start_year} AND {end_year}
    )
    SELECT *
    FROM year_partition
    WHERE rn <= {sample_size}
    ORDER BY pubdate ASC;""").fetch_df()
    # 先按照pubdate排序，再random排序


    else:
        # sample_size = 总量LIMIT 
        df = duck_conn.execute(f"""
    SELECT * 
    FROM papers
    WHERE abstract IS NOT NULL 
        AND abstract <> ''
        AND {pmid_filter}
        AND pubdate BETWEEN {start_year} AND {end_year}
    ORDER BY pubdate, RANDOM()
    LIMIT {sample_size}
        """).fetch_df()
        # 筛：不是空 + 不是空字符串(<>)
        # 返：LIMIT {sample_size} 指定返回的最大行数，{sample_size} 是一个变量，表示要返回的样本数量。
        # 转：fetch_df() 是 DuckDB 提供的一个方法，用于将 SQL 查询的结果直接转化为 Pandas DataFrame 对象，方便后续的数据分析和处理。

    

    print('* loaded sample data %s' % df.shape[0])


    #%% parse the software names of the given df
    # df.iterrows() ：用于逐行遍历 DataFrame。它返回一个生成器，每次迭代返回一个包含“行索引”和“行数据”的元组
    # df.shape ：返回一个包含 DataFrame 【行数和列数的元组】。df.shape[0] 获取 DataFrame 的行数
    # tqdm把df.iterrows()生成器（每次迭代返回两个值给i和row）包装到tqdm里边，并指定总长度total

    data = []
    total = df.shape[0]
    one_third_point = total//3
    all_abstract = [] # 所有文件的title + abstract，用于bert做inference，index一一对应
    all_other_info = [] # 用户huggingface上传，index一一对应
    for i, row in tqdm(df.iterrows(), total=total):
        # create a new paper for extraction
        all_abstract.append(row["title"].strip() + ". " + row["abstract"].strip())
        all_other_info.append((row['pmid'], row["pubdate"], \
                               row["journal"], row["mesh_terms"],\
                               row["authors"]))
    
    # --- 预处理，处理成txt文件 --- #
    taged_test_name = output_file + f"/Pubmed_{START_YEAR}_{END_YEAR}_{SAMPLE_SIZE}.txt"

    with open(taged_test_name, "w", encoding="utf-8") as taged: # Use writelines to write list 
        for item in all_abstract:
            cur_list = item.strip().split(" ")
            for cur_mention in cur_list:
                # BUG：必须要把O加上去，之后的逻辑是如果len!=2直接不处理
                if len(cur_mention) == 0:
                    continue
                if cur_mention[-1] in [",", ".", "!", "?"]:
                    taged.write(cur_mention[0:-1] + "\t" + 'O' + "\n")
                    taged.write(cur_mention[-1] + "\t" + 'O'+ "\n")
                else:
                    taged.write(cur_mention + "\t" + 'O' + "\n")
            taged.write("\n") 

    # 预处理后的目标文件为taged_test_name，空格分隔。返回每个abstract的BIO tag【完成后处理】
    # 对应的信息在all_other_info对应下标元素中，pmid等信息
 
    extracted_information = extract_and_save_to_db(taged_test_name)




    # --- data是为了huggingface构建，count是为了发送邮件 --- #
    data.append(current)
    if count >= one_third_point and count < one_third_point + 1:
        send_email(f"Llama 3.1 _ Software name Inference Progress: {start_year}_{end_year}_{sample_size}_shots:{shots_number} in each year", \
                    "Inference reached first one_third_point!")
    elif count >= 2*one_third_point and count < 2*one_third_point + 1:
        send_email(f"Llama 3.1 _ Software name Inference Progress: {start_year}_{end_year}_{sample_size}_shots:{shots_number} in each year", \
                    "Inference reached second one_third_point!")

    # in case sending too many requests
    # pause a few seconds every 100 requests
    # 使用openai api的时，为了防止请求速度过快而设置的sleep
    if i % 100 == 0: time.sleep(1)

    upload_hugging_face(data, start_year=start_year, end_year=end_year, each_year_sample_size=sample_size)
    send_email(f"Llama 3.1 _ Software name Inference Progress: {start_year}_{end_year}_{sample_size}_in each year ", \
               f"Inference finished and has been uploaded to 'YBXL/SWN_LLama3.1_{start_year}_{end_year}_{sample_size}'")

    print(f'* done! all papers are processed and saved into {db.path_db}')


#%% main function

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(
        description='Extract software names from academic papers',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        # 帮助格式化器，它会显示命令行参数的默认值。
    )
    def boolean_string(s):
        if s not in {'False', 'True'}:
            raise ValueError('Not a valid boolean string')
        return s == 'True'

    ## Test Required parameters
    parser.add_argument("--train_file", default=None, type=str)
    parser.add_argument("--eval_file", default=None, type=str)
    parser.add_argument("--test_file", default=None, type=str)
    parser.add_argument("--model_name_or_path", default=None, type=str)
    parser.add_argument("--output_dir", default="../output", type=str)

    ## Test other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--do_train", default=False, type=boolean_string)
    parser.add_argument("--do_eval", default=False, type=boolean_string)
    parser.add_argument("--do_test", default=False, type=boolean_string)
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--eval_batch_size", default=8, type=int)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--num_train_epochs", default=10, type=float)
    parser.add_argument("--warmup_proprotion", default=0.1, type=float)
    parser.add_argument("--use_weight", default=1, type=int)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", default=False)
    parser.add_argument("--loss_scale", type=float, default=0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--do_lower_case", action='store_true')
    parser.add_argument("--logging_steps", default=500, type=int)
    parser.add_argument("--clean", default=False, type=boolean_string, help="clean the output dir")
    parser.add_argument("--push_hf", default=False, type=boolean_string, help="clean the output dir")
    parser.add_argument("--need_birnn", default=False, type=boolean_string)
    parser.add_argument("--rnn_dim", default=128, type=int)

    
    # --- Test with db creation & filter --- #
    # 位置参数（不加两个--），必须要给
    parser.add_argument('--action', type=str, help='The action to perform: demo, extract', choices=['demo', 'extract'])
    # 可选参数，使用--之后才给，也可以不给
    parser.add_argument('--each_year_sample_size', type=int, default=10, help='The number of samples to extract')
    parser.add_argument('--pmid_filter', type=str, default='1=1', help='Other specific filter instrction')
    # 永真式，不需要就没用，需要再改
    parser.add_argument('--start_year', type=str, default='2009', help='The start year of papers to extract')
    parser.add_argument('--end_year', type=str, default='2023', help='The end year of papers to extract')
    parser.add_argument('--percent', type=float, default=1, help='Used X percentage from the resignated data')
    

    args = parser.parse_args()
    
    START_YEAR = args.start_year
    END_YEAR = args.end_year
    SAMPLE_SIZE = args.each_year_sample_size
    PERCENT = args.percent

    # 创建有年份的数据库名
    create_db(START_YEAR, END_YEAR, SAMPLE_SIZE, PERCENT)

    if args.action == 'extract':
        extract_and_save_samples(
            sample_size = args.each_year_sample_size,
            pmid_filter = args.pmid_filter,
            start_year = args.start_year,
            end_year = args.end_year,
            percent = args.percent
        )
    else:
        print('Unknown action: %s' % args.action)
        parser.print_help()
        exit(1)