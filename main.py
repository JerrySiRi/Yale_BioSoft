#%% load libs
import os
import sys
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

print('* loaded packages')


# %% load data
path_data = '/data/pubmed/metadata_36m.tsv'
df = pd.read_csv(
    path_data,
    sep='\t'
)

#%% init openai client
openai = OpenAI(
    api_key=os.getenv('OPENAI_APIKEY')
)

# %%
print('d %s' % d)