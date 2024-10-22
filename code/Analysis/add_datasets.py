from datasets import load_dataset
from datasets import Dataset, DatasetDict
from copy import deepcopy 
from huggingface_hub import HfApi, HfFolder
import os
import json
import random
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')

dataset_name = ["YBXL/SWN_LLama3.1_2019_2023_15000", "YBXL/SWN_LLama3.1_2014_2018_15000"]
new_name = ["JerrySiRi/SWN_Bert_based_2019_2023_15000", "JerrySiRi/SWN_Bert_based_2014_2018_15000"]
for index in range(0,len(dataset_name)):
    cur_data = dataset_name[index]
    dataset = load_dataset(cur_data, split='train')
    cur_name = cur_data[cur_data.find("/")+1:]
    cur_name = new_name[index]
    dataset.push_to_hub(cur_name, token=HF_TOKEN)