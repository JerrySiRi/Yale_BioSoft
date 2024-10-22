#%%
import os
import pandas as pd
from dotenv import load_dotenv


load_dotenv()
path_data = os.getenv('PUBMED_DATA_TSV')
df = pd.read_csv(path_data, encoding="utf-8", sep='\t')
print(df.head(-5))
#%%
print(df.head(50))
# %%
