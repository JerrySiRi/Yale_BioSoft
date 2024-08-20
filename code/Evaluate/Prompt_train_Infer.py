import os
import sys
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
from datasets import Dataset, DatasetDict
from ner_metrics_both import classification_report
from copy import deepcopy
import tqdm


# -- 加入llama3.1的infer模块 -- #
load_dotenv()
LLama31_infer_path = os.getenv('LLama31_infer_path')
sys.path.append(LLama31_infer_path)
from llama_infer import *

# -- 加入gpt-4o的infer模块
load_dotenv()
gpt_infer_path = os.getenv('GPT_infer_path')
sys.path.append(gpt_infer_path)
from Inference_pubmed import *


# -- 加入gpt-4o的infer代码 -- #



# 不做删除，直接占位，之后给成O，保持和ann处理之后的长度一致！

def process_special_token(orignial_list):
    new_list = deepcopy(orignial_list)

    count = 0
    for index in range(0,len(orignial_list)-1):
        if index in [0,1,2]:
            continue
        else:
            if (orignial_list[index] == "-") or (orignial_list[index] == "/"):
                new_list[index-1-count] = orignial_list[index-1]+orignial_list[index]+orignial_list[index+1]
                del new_list[index-count]
                del new_list[index-count]
                count += 2
            elif orignial_list[index] == "(":
                new_len = 0 # 专门为括号而生
                new_str = str()

                while(index+new_len<len(orignial_list)-1 and orignial_list[index+new_len]!=")"):
                    new_len += 1

                # 最后额外多加了一次1，会出现问题，if判断一下吧
                if index+new_len+1 < len(orignial_list):
                    new_str = ("").join(orignial_list[index:index+new_len+1])
                else:
                    new_str = ("").join(orignial_list[index:len(orignial_list)])
                    continue

                new_list[index-count] = new_str
                for _ in range(new_len):
                    del new_list[index-count+1]
                count += new_len

    return new_list
            


# --- 子列表匹配，返回匹配的首、尾对应的元组列表 --- #
def find_sublist(main_list, sub_list):
    sub_len = len(sub_list)
    index_list = list()
    flag_in = False
    for i in range(len(main_list) - sub_len + 1):
        #if main_list[i:i + sub_len] == sub_list:
        if ("").join(sub_list) in ("").join(main_list[i:i + sub_len]):
            # 解决可能有分割不准确的情况，子列表->子字符串匹配
            index_list.append((i,i+sub_len-1))
            flag_in = True
    if flag_in == True: # 找到了
        return index_list
    else: # 没找到 
        return -1  



# ------ ann文件结果生成BIO tag，为了评估 ----- # 
def convert_ann_to_bio(text, entities):
    gold_cant_match = 0

    text = text.lower()
    no_spaces_text = text.split(" ")
    no_spaces_text = process_special_token(no_spaces_text)

    software_names = list(set(entities)) # 文中可能会多次复现，去重
    software_names_list = [process_special_token(item.lower().split(" ")) for item in software_names]

    bio_index = list()
    for item in software_names_list:
        current_index_list = find_sublist(no_spaces_text, item)
        if current_index_list == -1:
            print(no_spaces_text)
            print("ann to bio","*"*10, item)
            gold_cant_match += 1

        else:
            bio_index += current_index_list

    bio_list = ["O"] * len(no_spaces_text)
    # 遍历每个索引对，并设置对应位置为True
    for start, end in bio_index:
        for i in range(start, end + 1):  # 因为end是包含的，所以用end + 1
            if i == start:
                bio_list[i] = "B"
            else:
                bio_list[i] = "I"
    return bio_list, gold_cant_match
    



# ------- llama3.1输出结果生成BIO tag，为了评估 ------- #
# input: 原始文本, 抽取结果
# output: ['O', 'B-PER', 'O', 'B-ORG', 'B-ORG', 'I-ORG', 'O', 'B-PER', 'I-PER', 'O']
def convert_txt_to_bio(text, entities):

    # 原始文本全部变成小写 & 把“-”前后给合并了
    text = text.lower()
    no_spaces_text = text.split(" ")
    no_spaces_text = process_special_token(no_spaces_text)

    # softwarename变成[[],[]]的格式，做子列表的匹配
    software_names = list(set([item["name"] for item in entities]))
    # print("*"*5, software_names)
    software_names_list = [process_special_token(item.lower().split(" ")) for item in software_names]
    # print("#"*5, software_names_list)

    bio_index = list()
    for item in software_names_list:
        # 子列表的匹配
        current_index_list = find_sublist(no_spaces_text, item)
        if current_index_list == -1:
            
            print(no_spaces_text)
            print("txt to bio","="*10, item)
        else:
            bio_index += current_index_list

    bio_list = ["O"] * len(no_spaces_text)
    # 遍历每个索引对，并设置对应位置为True
    for start, end in bio_index:
        for i in range(start, end + 1):  # 因为end是包含的，所以用end + 1
            if i == start:
                bio_list[i] = "B"
            else:
                bio_list[i] = "I"

    return bio_list








# ---------- 调用gpt-4o和llama3.1来推理 -------- # 

def gpt_4o_infer(system_role, prompt_template, paper):
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
        # 共用llama3.1相同的guidelines和few_shots
        Prompt_all = f"""# You are given a title and an abstract of an academic publication. Your task is to identify and extract the names of software mentioned in the abstract. Software names are typically proper nouns and may include specific tools, platforms, or libraries used in research. Please list the software names you find in the publication in a JSON object using a key "software". If you are unable to identify any software names, please return an empty list of "software". When identifying software names, please consider the following exclusion criteria 
                            Also, apply following \"Guidelines\" and refer following \"Gold Examples\" to help with accuracy \n"""\
                            f"# Guidelines: {guidelines} \n"\
                            f"# Gold Examples: {few_shots} \n"\
                            f"# INPUT: {prompt} \n"\
                            f"\n"\
                            f"# OUTPUT: \n"
        
        client = OpenAI(api_key = OPENAI_KEY)
        completion = client.chat.completions.create(
            model = "gpt-4o-mini",
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





def llama_31_infer(paper):
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
    return result
    
    




if __name__ == "__main__":

    # 训练+测试集数据
    train_folder_path = "../../datasets/train_data"
    test_folder_path = "../../datasets/test_gold"
    train_files = sorted(os.listdir(train_folder_path))
    test_files = sorted(os.listdir(test_folder_path))


    # 所有的train+test data
    train_path="../../datasets/train_data/"
    test_path="../../datasets/test_gold/"
    train_txt_files = sorted([train_path + f for f in train_files if f.endswith('.txt')])
    test_txt_files = sorted([test_path + f for f in test_files if f.endswith('.txt')])
    train_ann_files = sorted([train_path + f for f in train_files if f.endswith('.ann')])
    test_ann_files = sorted([test_path + f for f in test_files if f.endswith('.ann')])

    new_txt = train_txt_files + test_txt_files
    new_ann = train_ann_files + test_ann_files
    
    # 【目前按照llama3.1的格式来写，gpt-4o需要修改】
    all_gold = list()
    llama_all_pred = list()
    gpt4o_all_pred = list()

    merged_list = zip(new_txt, new_ann)
    gold_cant_match = 0
    for txt, ann in tqdm(merged_list, total=len(new_ann)):
        current_paper = dict() # 包含pmid, title 和 abstract的字典

        file_name = os.path.basename(txt)  
        file_name_without_extension = os.path.splitext(file_name)[0]  # 去除扩展名
        # 此时abstract和title先分开放，到时候调llama3.1推理的时候再合并

        with open(txt, "r", encoding="utf-8") as x_data: # Use writelines to write list
            count = -1
            current_paper["pmid"] = file_name_without_extension
            for line in x_data:
                count += 1
                processed_line = line.strip()
                if count==0:
                    current_paper["title"] = processed_line
                    current_paper["abstract"] = ""
                else:
                    current_paper["abstract"] += processed_line
        
        gold_entities = list()
        with open(ann, "r", encoding="utf-8") as y_data: # Use writelines to write list
            for line in y_data:
                current_mes = line.strip().split("\t")
                gold_entities.append(current_mes[-1])


        
        pred_llama_entities = llama_31_infer(current_paper) # 在函数内把current_paper给改了，把title加到了abstract上
        pred_gpt4o_entities = gpt_4o_infer(current_paper) # 在函数内把current_paper给改了，把title加到了abstract上
        
        pred_llama_bio_list = convert_txt_to_bio(current_paper["abstract"], pred_llama_entities["software"])
        pred_gpt4o_bio_list = convert_txt_to_bio(current_paper["abstract"], pred_gpt4o_entities["software"])

        llama_all_pred.append(pred_llama_bio_list)
        gpt4o_all_pred.append(pred_gpt4o_bio_list)
        
        gold_bio_list, current_count = convert_ann_to_bio(current_paper["abstract"], gold_entities)
        gold_cant_match += current_count # 真实entity匹配补上的
        all_gold.append(gold_bio_list)

    llama_metrics = classification_report(tags_true=gold_bio_list, tags_pred=pred_llama_bio_list, mode="lenient") # for lenient match
    gpt4o_metrics = classification_report(tags_true=gold_bio_list, tags_pred=pred_gpt4o_bio_list, mode="lenient") # for lenient match
    # "lenient", "strict"
    print("The result of LLaMA3.1 is", llama_metrics)
    print("The result of Gpt-4o is", gpt4o_metrics)
    
        
        
