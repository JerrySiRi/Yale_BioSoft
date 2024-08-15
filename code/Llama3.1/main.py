#%% run demo
# import openai
import json
import time
import os
from tqdm import tqdm
import pandas as pd
import duckdb
import pysbd
from openai import OpenAI
import sqlite3
import db
from pprint import pprint

from dotenv import load_dotenv
load_dotenv()
print('* loaded configs')
print('* OPENAI_API_BASE_URL=%s' % os.getenv('OPENAI_API_BASE_URL'))
print('* OPENAI_API_MODEL=%s' % os.getenv('OPENAI_API_MODEL'))
print('* PUBMED_DATA_TSV=%s' % os.getenv('PUBMED_DATA_TSV'))
print('* OUTPUT_DATABASE=%s' % os.getenv('OUTPUT_DATABASE'))

print('* loaded all libraries')


#%% load openai client
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


TPL_PROMPT = """You are given a title and an abstract of an academic publication. Your task is to identify and extract the names of software mentioned in the abstract. Software names are typically proper nouns and may include specific tools, platforms, or libraries used in research. Please list the software names you find in the publication in a JSON object using a key "software". If you are unable to identify any software names, please return an empty list of "software". When identifying software names, please consider the following exclusion criteria:

- Exclude general terms that are not specific software names (e.g., "software", "tool", "platform", "system").
- Exclude common programming languages (e.g., "Python", "Java", "R", "Julia", "Rust", "Golang", "PHP").
- Exclude common web technologies (e.g., "HTML", "CSS", "JavaScript").
- Exclude websites and cloud platforms (e.g., "GitHub", "Google Cloud", "Amazon Web Services").

The publication details are as follows:

Title: {title}
     
Abstract: {abstract}
"""

SYSTEM_ROLE = "You are an experienced software developer, data scientist, and researcher in biomedical fields, skilled in developing software using various techniques."

def extract(system_role, prompt_template, paper):
    '''
    Extract something from the abstract of a paper based on the given prompt template.
    '''
    try:
        prompt = prompt_template.format(**paper)

        completion = client.chat.completions.create(
            model = os.getenv("OPENAI_API_MODEL"),
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": prompt}
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
sent_segmenter = pysbd.Segmenter(language="en", clean=False)


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
    sents = sent_segmenter.segment(abstract)
    
    for entity in entities:
        # search this entity in all sentences
        # this entity may appear in multiple sentences
        contexts = []
        for sent in sents:

            # the entity should be a substring of the sentence
            # but sometimes the entity may be in different forms
            # e.g., "MetaMap" vs. "metamap"
            # so we use lower case for comparison
            if entity.lower() in sent.lower():
                contexts.append(sent)

        ents.append({
            'name': entity,
            'contexts': contexts
        })
                
    return ents


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
            print(f'! error: {e}')
            print(f'! failed to get context for {tmp}')
            software_names_with_contexts = []

        result = {'software': software_names_with_contexts}
    
    db.save_paper_software_names(paper['pmid'], result)


def demo_extract():
    '''
    Run a demo to extract software names from the sample paper.
    '''
    sample_paper = {
        "pmid": "35613942",
        "title": "A Systematic Approach to Configuring MetaMap for Optimal Performance",
        "abstract": """Background: MetaMap is a valuable tool for processing biomedical texts to identify concepts. Although MetaMap is highly configurative, configuration decisions are not straightforward.

    Objective: To develop a systematic, data-driven methodology for configuring MetaMap for optimal performance.

    Methods: MetaMap, the word2vec model, and the phrase model were used to build a pipeline. For unsupervised training, the phrase and word2vec models used abstracts related to clinical decision support as input. During testing, MetaMap was configured with the default option, one behavior option, and two behavior options. For each configuration, cosine and soft cosine similarity scores between identified entities and gold-standard terms were computed for 40 annotated abstracts (422 sentences). The similarity scores were used to calculate and compare the overall percentages of exact matches, similar matches, and missing gold-standard terms among the abstracts for each configuration. The results were manually spot-checked. The precision, recall, and F-measure (β =1) were calculated.

    Results: The percentages of exact matches and missing gold-standard terms were 0.6-0.79 and 0.09-0.3 for one behavior option, and 0.56-0.8 and 0.09-0.3 for two behavior options, respectively. The percentages of exact matches and missing terms for soft cosine similarity scores exceeded those for cosine similarity scores. The average precision, recall, and F-measure were 0.59, 0.82, and 0.68 for exact matches, and 1.00, 0.53, and 0.69 for missing terms, respectively.

    Conclusion: We demonstrated a systematic approach that provides objective and accurate evidence guiding MetaMap configurations for optimizing performance. Combining objective evidence and the current practice of using principles, experience, and intuitions outperforms a single strategy in MetaMap configurations. Our methodology, reference codes, measurements, results, and workflow are valuable references for optimizing and configuring MetaMap.
    """
    }
    ret = extract(SYSTEM_ROLE, TPL_PROMPT, sample_paper)
    print("* extracted software names:", ret)

    if ret is None:
        pass
    else:
        entities_with_context = get_contexts(ret['software'], sample_paper)
        print('* entities with context:')
        pprint(entities_with_context)



def extract_and_save_samples(
    sample_size=10, 
    pmid_filter='pmid > 34500000'
):
    '''
    Extract software names from all papers in the database.
    '''
    print(f'* extracting software names of {sample_size} papers where {pmid_filter}')
    #%% load the data from 
    duck_conn = duckdb.connect()

    # load data
    path_data = os.getenv('PUBMED_DATA_TSV')

    print('* loading data from %s (whole pubmed data is about 50G, it may take a few minutes)' % path_data)
    duck_conn.execute(f"""
        CREATE TABLE IF NOT EXISTS papers AS
        SELECT *
        FROM read_csv_auto('{path_data}', delim='\\t', header=True)
    """)
    print('* loaded data from %s' % path_data)

    # make a small sample
    df = duck_conn.execute(f"""
SELECT * 
FROM papers
WHERE abstract IS NOT NULL AND abstract <> ''
    AND {pmid_filter}
ORDER BY RANDOM()
LIMIT {sample_size}
    """).fetch_df()

    print('* loaded sample data %s' % df.shape[0])


    #%% parse the software names of the given df
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        # create a new paper for extraction
        paper = {
            'pmid': row['pmid'],
            'title': row['title'],
            'abstract': row['abstract']
        }
        
        extract_and_save_to_db(paper)

        # in case sending too many requests
        # pause a few seconds every 100 requests
        if i % 100 == 0: time.sleep(1)

    print(f'* done! all papers are processed and saved into {db.path_db}')


#%% main function
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Extract software names from academic papers',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('action', type=str, help='The action to perform: demo, extract', choices=['demo', 'extract'])
    parser.add_argument('--sample_size', type=int, default=10, help='The number of samples to extract')
    parser.add_argument('--pmid_filter', type=str, default='pmid > 34500000', help='The filter to select papers')

    args = parser.parse_args()

    if args.action == 'demo':
        demo_extract()

    elif args.action == 'extract':
        extract_and_save_samples(
            args.sample_size,
            args.pmid_filter
        )

    else:
        print('Unknown action: %s' % args.action)
        parser.print_help()
        exit(1)