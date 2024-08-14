import os
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
from datasets import Dataset, DatasetDict
# ---------- 测试prompt：用本地的train数据来infer -------- # 




if __name__ == "__main__":

    # 把原始的(x,y)
    folder_path = "../../datasets/train_data"
    files = sorted(os.listdir(folder_path))

    txt_files = [f for f in files if f.endswith('.txt')]
    ann_files = [f for f in files if f.endswith('.ann')]
    
    # ann_files = [f for f in files if f.endswith('.ann')]

    for initial_text in txt_files:
        with open(initial_text, "w", encoding="utf-8") as x_data: # Use writelines to write list
            