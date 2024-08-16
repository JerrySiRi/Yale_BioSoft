#%% run demo
# import openai
import json
import time
import os
import sys
import sqlite3
import db
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



#%% load openai client

# 现在用的本地的llama3.1，只不过名字没改
base_url = os.getenv("OPENAI_API_BASE_URL", None)

if base_url is None or base_url == '':
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    print('* loaded official openai client')
else:
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=base_url
    )
    print('* loaded custom openai client at %s' % base_url)



# ----------------------- 读取设计好的guidelines ----------------- # 
guidelines = """"""
few_shots = """"""
with open("../../datasets/prompts/guidelines.txt", 'r', encoding='utf-8') as file_txt:
    for line in file_txt:
        guidelines += line
with open("../../datasets/prompts/few_shots_Llama31.txt", 'r', encoding='utf-8') as file_txt:
    for line in file_txt:
        few_shots += line

TPL_PROMPT = """

Title: {title}
     
Abstract: {abstract}
"""

# 这个会比annotator要好！
SYSTEM_ROLE = "You are an experienced software developer, data scientist, and researcher in biomedical fields, skilled in developing software using various techniques and particularly well-versed in the names of software used in this domain."


# --- 调用model，进行inference --- #
def extract(system_role, prompt_template, paper):
    '''
    Extract something from the abstract of a paper based on the given prompt template.
    '''
    try:
        # 传入的是TPL_prompt, 里边有format函数要用的{title}和{abstract}。
        # 传入的paper会给键值对
        # 【改】把原来的prompt加到了abstract里边
        paper["abstract"] = paper["title"] + paper["abstract"]
        prompt = prompt_template.format(**paper)

        # 返回的是一个json对象，有"software"关键字

        Prompt_all = f"""# You are given a title and an abstract of an academic publication. Your task is to identify and extract the names of software mentioned in the abstract. Software names are typically proper nouns and may include specific tools, platforms, or libraries used in research. Please list the software names you find in the publication in a JSON object using a key "software". If you are unable to identify any software names, please return an empty list of "software". When identifying software names, please consider the following exclusion criteria 
                            Also, apply following \"Guidelines\" and refer following \"Gold Examples\" to help with accuracy \n"""\
                            f"# Guidelines: {guidelines} \n"\
                            f"# Gold Examples: {few_shots} \n"\
                            f"# INPUT: {prompt} \n"\
                            f"\n"\
                            f"# OUTPUT: \n"

        completion = client.chat.completions.create(
            model = os.getenv("OPENAI_API_MODEL"),
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": Prompt_all}
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )

        result = json.loads(completion.choices[0].message.content)

        return result
    
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



# --- 单篇文章 抽取 & 保存到数据库 --- #
def extract_and_save_to_db(paper, flag_force_update=False):
    '''
    Extract software names from the given paper and save the result to the database.
    
    The paper should be a dictionary with the following keys
    - pmid: the PMID of the paper
    - title: the title of the paper
    - abstract: the abstract of the paper

    '''
    # check if the software names are already extracted
    result = db.load_paper_software_names(paper['pmid'])

    # 如果这篇文章（pmid）已经有软件名了（之前提取过），flag=False时啥都不做，True时删掉重新提取这篇文章的软件名
    if result is not None:
        if flag_force_update:
            # delete the existing software names
            db.delete_paper_software_names(paper['pmid'])
            # then extract again
        else:
            # already extracted
            print(f'* found software for paper {paper["pmid"]}')
            return

    # not found, extract the software names
    # extract using the openai model
    tmp = extract(SYSTEM_ROLE, TPL_PROMPT, paper)

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
    
    db.save_paper_software_names(paper['pmid'], paper["pubdate"], result)

    current = dict()
    current["pmid"] = paper['pmid']
    current["pubdate"] = paper['pubdate']
    current["software"] = result["software"]
    return current



def demo_extract():
    '''
    Run a demo to extract software names from the sample paper.
    '''
    # 问题：漏掉了title的抽取，应该把title也放到abstract里边
    sample_paper_initial = {
        "pmid": "35613942",
        "title": "A Systematic Approach to Configuring MetaMap for Optimal Performance",
        "abstract": """Background: MetaMap is a valuable tool for processing biomedical texts to identify concepts. Although MetaMap is highly configurative, configuration decisions are not straightforward.

    Objective: To develop a systematic, data-driven methodology for configuring MetaMap for optimal performance.

    Methods: MetaMap, the word2vec model, and the phrase model were used to build a pipeline. For unsupervised training, the phrase and word2vec models used abstracts related to clinical decision support as input. During testing, MetaMap was configured with the default option, one behavior option, and two behavior options. For each configuration, cosine and soft cosine similarity scores between identified entities and gold-standard terms were computed for 40 annotated abstracts (422 sentences). The similarity scores were used to calculate and compare the overall percentages of exact matches, similar matches, and missing gold-standard terms among the abstracts for each configuration. The results were manually spot-checked. The precision, recall, and F-measure (β =1) were calculated.

    Results: The percentages of exact matches and missing gold-standard terms were 0.6-0.79 and 0.09-0.3 for one behavior option, and 0.56-0.8 and 0.09-0.3 for two behavior options, respectively. The percentages of exact matches and missing terms for soft cosine similarity scores exceeded those for cosine similarity scores. The average precision, recall, and F-measure were 0.59, 0.82, and 0.68 for exact matches, and 1.00, 0.53, and 0.69 for missing terms, respectively.

    Conclusion: We demonstrated a systematic approach that provides objective and accurate evidence guiding MetaMap configurations for optimizing performance. Combining objective evidence and the current practice of using principles, experience, and intuitions outperforms a single strategy in MetaMap configurations. Our methodology, reference codes, measurements, results, and workflow are valuable references for optimizing and configuring MetaMap.
    """
    }
    sample_paper_gold = {
        "pmid": "000000",
        "title":"i - ADHoRe 2.0 : an improved tool to detect degenerated genomic homology using genomic profiles .",
        "abstract":
    """ 
SUMMARY : i - ADHoRe is a software tool that combines gene content and gene order information of homologous genomic segments into profiles to detect highly degenerated homology relations within and between genomes .
The new version offers , besides a significant increase in performance , several optimizations to the algorithm , most importantly to the profile alignment routine .
As a result , the annotations of multiple genomes , or parts thereof , can be fed simultaneously into the program , after which it will report all regions of homology , both within and between genomes .
AVAILABILITY : The i - ADHoRe 2.0 package contains the C + + source code for the main program as well as various Perl scripts and a fully documented Perl API to facilitate post - processing .
The software runs on any Linux - or - UNIX based platform .
The package is freely available for academic users and can be downloaded from http : / / bioinformatics.psb.ugent.be / 
    """
    }
    ret = extract(SYSTEM_ROLE, TPL_PROMPT, sample_paper_initial)
    print("* extracted software names:", ret)

    if ret is None:
        pass
    else:
        entities_with_context = get_contexts(ret['software'], sample_paper_initial)
        print('* entities with context:')
        pprint(entities_with_context)



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

    dataset_dict.push_to_hub(f'YBXL/SWN_LLama3.1_{start_year}_{end_year}_{each_year_sample_size}', token=HF_TOKEN)


# --- 总抽取函数 --- # 

def extract_and_save_samples(
    sample_size=10, 
    pmid_filter='1=1',
    start_year = 2010,
    end_year = 2023,
    main_run = True,
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
    halfway = total//2
    count = 0
    for i, row in tqdm(df.iterrows(), total=total):
        count += 1
        # create a new paper for extraction
        paper = {
            'pmid': row['pmid'],
            'title': row['title'],
            'pubdate': row["pubdate"],
            'abstract': row['abstract']
        }
        
        current = extract_and_save_to_db(paper)
        data.append(current)
        if count >= halfway and count < halfway + 1:
            send_email(f"Llama 3.1 _ Software name Inference Progress: {start_year}_{end_year}_{sample_size}_in each year", \
                       "Inference reached halfway point!")

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
    # 位置参数，必须要给
    parser.add_argument('action', type=str, help='The action to perform: demo, extract', choices=['demo', 'extract'])
    # 可选参数，使用--之后才给，也可以不给
    parser.add_argument('--each_year_sample_size', type=int, default=10, help='The number of samples to extract')
    parser.add_argument('--pmid_filter', type=str, default='1=1', help='Other specific filter instrction')
    # 永真式，不需要就没用，需要再改
    parser.add_argument('--start_year', type=str, default='2009', help='The start year of papers to extract')
    parser.add_argument('--end_year', type=str, default='2023', help='The end year of papers to extract')
    

    args = parser.parse_args()

    if args.action == 'demo':
        demo_extract()

    elif args.action == 'extract':
        extract_and_save_samples(
            sample_size = args.each_year_sample_size,
            pmid_filter = args.pmid_filter,
            start_year = args.start_year,
            end_year = args.end_year
        )

    else:
        print('Unknown action: %s' % args.action)
        parser.print_help()
        exit(1)