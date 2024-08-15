# --------------------- 把推理格式<mark> -> 转换成.ann格式，方便用data_preprocess的方法 ---------------- #  


from openai import OpenAI
import os
from dotenv import load_dotenv
import pandas as pd
from datasets import load_dataset

def mark_extract(text):
    pass



if __name__ == "__main__":
    infered_data = load_dataset("YBXL/STN_4shots_1k")["test"]
    

