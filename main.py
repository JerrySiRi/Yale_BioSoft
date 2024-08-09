#%% load libs
import os
import sys
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

print('* loaded packages')


# %% load data
try:
    path_data = os.getenv('PATH_TO_36M_TSV')

    print('* loading something from %s' % path_data)
    df = pd.read_csv(
        path_data,
        sep='\t'
    )
    print('* loaded all the data from %s' % path_data)

except:
    pass

#%% init openai client
openai = OpenAI(
    api_key=os.getenv('OPENAI_APIKEY')
)
print('* inited openai client')

# %%
print('* done!')
