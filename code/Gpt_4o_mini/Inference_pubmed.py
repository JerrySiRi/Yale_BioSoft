# ---------------- 测试好了prompt，预测pubmed中的数据 -------------- #

# -------------------- read data ------------------ #
from openai import OpenAI
import os
from dotenv import load_dotenv
import pandas as pd
from datasets import Dataset, DatasetDict




# ----------------- openai_in-contenxt learning -------------------- # 


def read_guidelines_shots(prompts_foler_path, guidelines_name, few_shots_name):
    guidelines = """"""
    few_shots = """"""
    with open(prompts_foler_path+"/"+guidelines_name, 'r', encoding='utf-8') as file_txt:
        for line in file_txt:
            guidelines += line
    with open(prompts_foler_path+"/"+few_shots_name, 'r', encoding='utf-8') as file_txt:
        for line in file_txt:
            few_shots += line
    return guidelines, few_shots
        


# 【【Chat Completion -- 对话补全】】

def Inference_based_on_text(text, OPENAI_KEY = None, guidelines=None, few_shots=None):
    client = OpenAI(api_key = OPENAI_KEY)
    current_prompt = [
        {"role": "system", 
        "content": """You are an experienced annotator with extensive knowledge in the field of biomedical informatics and data science, particularly well-versed in the names of software used in this domain"""},

        {"role": "user", 
        "content": f"""Given an abstract identity all software names from it by highlighting within <mark> and </mark> tags. 
                    Also, apply following \"Guidelines\" and refer following \"Gold Examples\" to help with accuracy \n"""\
                    f"Guidelines: {guidelines} \n"\
                    f"Gold Examples: {few_shots} \n"\
                    f"INPUT: {text} \n"\
                    f"\n"\
                    f"OUTPUT: \n"
        }    
        #f"True Differential Diagnosis: {true_diagnosis_array}"
    ]
    completion = client.chat.completions.create(
    model="gpt-4o",
    messages = current_prompt
    )
    answer = completion.choices[0].message.content
    return answer, current_prompt




if __name__ == "__main__":

    prompts_foler_path = "../../datasets/prompts"
    guidelines_name = "guidelines.txt"
    few_shots_name = "few_shots.txt"

   # 加载 .env 文件中的环境变量
    load_dotenv()



    # 现在可以使用 os.getenv() 或 os.environ 获取变量
    OPENAI_KEY = os.getenv('OPENAI_APIKEY')
    path_data = os.getenv('PATH_TO_36M_TSV')

    path_data = '/data/pubmed/metadata_36m.tsv'

    try:
        print('* loading something from %s' % path_data)
        df = pd.read_csv(
            path_data,
            sep='\t'
        )
        print('* loaded all the data from %s' % path_data)
    except Exception as err:
        print("!!!", err)


    # ---------------------- 数据筛选 ------------------- # 
    # df是读出来的内容
    # 筛选abstract不是Nan的
    print("* Begin filtering")
    df_filtered = df.dropna(subset=["abstract"])

    df_infer = df_filtered[["pmid", "abstract"]]
    df_infer_1k = df_infer.head(1024) 
    number = 1*1024

    print("* Finish filtering & select first 1024 examples")   

    # df_infer_8 = df_infer.head(8)   
    data = []
    guidelines, few_shots = read_guidelines_shots(prompts_foler_path, guidelines_name, few_shots_name)

    # ---------------------  GPT-4o推理 ---------------------- # 
    print("* Start inference")
    count = 0
    for id, text in zip(df_infer_1k["pmid"], df_infer_1k["abstract"]):
        current_data = dict()
        current_data["id"] = id
        answer, current_prompt = Inference_based_on_text(text,OPENAI_KEY=OPENAI_KEY, guidelines=guidelines, few_shots=few_shots)
        current_data["query"] = current_prompt
        current_data["answer"] = answer
        data.append(current_data)
        count += 1
        if count == 0.5*number:
            print("* Half of inference has finished")
    print("* Finish inference")

    # ---------------- 上传到huggingface中 -------------- #
    
    
    HF_TOKEN = "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"
    print("* Huggingface token", HF_TOKEN)
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

    dataset_dict.push_to_hub('YBXL/STN_8shots_1k', token=HF_TOKEN)



