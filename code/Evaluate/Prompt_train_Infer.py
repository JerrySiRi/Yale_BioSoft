#%%
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
OPENAI_APIKEY = os.getenv("OPENAI_APIKEY")
sys.path.append(gpt_infer_path)
from Inference_pubmed import *


# -- 加入gpt-4o的infer代码 -- #



# 不做删除，直接占位，之后给成O，保持和ann处理之后的长度一致！

def process_special_token(orignial_list):
    """
    处理gpt/llama3.1的输出，保持原有text不变
    input: [[],[],[]]
    TODO: 如果最里边的[]。两边加空格：-,/。左边加空格：!,+。拆开：（）
    output: 直接在原列表上修改
    """
    return_list = deepcopy(orignial_list)
    for entity in orignial_list:
        # 每一个item是一个列表
        dup_entity = deepcopy(entity)
        count = 0 # 删除的个数

        # each_index 是entity的坐标索引
        # dup_index = each_index-count 是dup_entity的坐标索引

        for each_index in range(0,len(entity)):
            dup_index = each_index - count

            if each_index == 0:
                continue

            # 有["a", "-", "b"], ["a", "/", "b"]。加入["a-b"]
            if (entity[each_index] == "-") or (entity[each_index] == "/"):
                dup_entity[dup_index-1] = dup_entity[dup_index-1]+dup_entity[dup_index]+dup_entity[dup_index+1]
                del dup_entity[dup_index]
                del dup_entity[dup_index]
                count += 2
                return_list.append(dup_entity)

            """
            # 有["a-b"]。加入["a","-","b"]
            elif ("-" in entity[each_index]) or ("/" in entity[each_index]):
                index_1 = entity[each_index].find("-")
                index_2 = entity[each_index].find("/")

                index = 0
                if index_1 == -1: index = index_2
                else: index = index_1

                if index < len(dup_entity[dup_index])-1:
                    dup_entity[dup_index] = dup_entity[dup_index][0:index]
                    dup_entity.insert(dup_index+1, dup_entity[dup_index][index+1:])
                    print(index, len(dup_entity[dup_index]))
                    dup_entity.insert(dup_index+1, dup_entity[dup_index][index])
                    count -= 2 
                    return_list.append(dup_entity)
            
            """
            
    return return_list
            



    
            


# --- 子列表匹配，返回匹配的首、尾对应的元组列表 --- #
def find_sublist(main_list, sub_list):
    """
    对处理后的gpt/llama3.1的输出，把原来的text和每一个entity做匹配
    """
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
    


# ------- llama3.1输出结果生成BIO tag，为了评估 ------- #
# input: 原始文本, 抽取结果
# output: ['O', 'B-PER', 'O', 'B-ORG', 'B-ORG', 'I-ORG', 'O', 'B-PER', 'I-PER', 'O']
def convert_txt_to_bio(text, entities):

    """
    逻辑得修改，是对entity做preprocess，而不是对text做啦。保证列表长度一致
    """
    # 原始文本全部变成小写
    text = text.lower()
    no_spaces_text = text.split(" ")

    # software name变成[[],[]]的格式，做子列表的匹配
    software_names = list(set([item["name"] for item in entities]))
    # print("*"*5, software_names)
    software_names_list = [item.lower().split(" ") for item in software_names]
    software_names_list = process_special_token(software_names_list)
    # print("#"*5, software_names_list)

    bio_index = list()
    for item in software_names_list:
        # 子列表的匹配
        current_index_list = find_sublist(no_spaces_text, item)
        if current_index_list == -1:
            print("txt to bio","="*10, item)
        else:
            bio_index += current_index_list

    bio_list = ["O"] * len(no_spaces_text)

    # 遍历每个索引对，并设置对应位置为B或者I
    for start, end in bio_index:
        for i in range(start, end + 1):  # 因为end是包含的，所以用end + 1
            if i == start:
                bio_list[i] = "B-SWN"
            else:
                bio_list[i] = "I-SWN"
    return bio_list



# ---------- 调用gpt-4o和llama3.1来推理 -------- # 

def gpt_4o_infer(paper):

    '''
    Extract something from the abstract of a paper based on the given prompt template.
    '''
    print("* gpt-4o is inferring")
    system_role = SYSTEM_ROLE
    prompt_template = TPL_PROMPT
    guidelines = """"""
    few_shots = """"""
    with open("../../datasets/prompts/guidelines.txt", 'r', encoding='utf-8') as file_txt:
        for line in file_txt:
            guidelines += line
    with open("../../datasets/prompts/few_shots_Llama31.txt", 'r', encoding='utf-8') as file_txt:
        for line in file_txt:
            few_shots += line

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
        
        client = OpenAI(api_key = OPENAI_APIKEY)
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
    print("* llama 3.1 is inferring")
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
    
    


#%%


#%%
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


        # ------ txt文件 ------ #
        txt_with_index_content = []
        with open(txt, "r", encoding="utf-8") as x_data: # Use writelines to write list
            
            count = -1
            current_paper["pmid"] = file_name_without_extension
            for line in x_data:
                # -- 让model infer的 -- #
                count += 1
                processed_line = line.strip()
                if count==0:
                    current_paper["title"] = processed_line
                    current_paper["abstract"] = ""
                else:
                    current_paper["abstract"] += " "+ processed_line
                    # BUG!!!!!! 分句的时候会去掉空格！导致最后长度不对
                
                # -- 让ann生成gold bio list的 -- #
                content = line.strip().split(" ")
                cur_index = 0
                for item in content:
                    txt_with_index_content.append((cur_index, item))
                    cur_index = cur_index + len(item) + 1

        # ------ ann文件 ------ #
        current_gold_list = list()
        current_gold_dict = dict()

        with open(ann, "r", encoding="utf-8") as y_data: # Use writelines to write list
            for line in y_data:
                current_mes = line.strip().split("\t")
                
                # 可能读到ann是空的 
                if len(current_mes) < 3:
                    continue
                
                # ann非空 
                index_initial = current_mes[1].strip().split(" ")
                if len(index_initial) == 3:
                    index = (int(index_initial[1]), int(index_initial[2]))
                elif len(index_initial) == 4: 
                    item_index = index_initial[2].find(";")
                    index = (int(index_initial[1]), int(index_initial[2][0:item_index]))
                current_gold_list.append((index, current_mes[-1]))

                current_gold_dict[index[0]] = (index[1], current_mes[-1])
                # index[0]是entity刚开始的坐标，index[1]是entity结束的坐标, current_mes是mention的名字
        

        # ------ 生成ann文件的gold label ------ #
        gold_current_label = list()
        flag = False
        last_index = 0
        for item in txt_with_index_content:
            men_index = item[0]
            men_name = item[1]
            if (flag == False) & (men_index in current_gold_dict.keys()): # matched! & B
                gold_current_label.append("B-SWN")
                flag = True
                last_index = current_gold_dict[men_index][0]
            elif flag == True: # matched! & I
                if men_index <= last_index:
                    gold_current_label.append("I-SWN")
                else: # end match & O
                    gold_current_label.append("O")
                    flag = False
            else:
                gold_current_label.append("O")
                


        # -- 两模型infer -- #
        pred_llama_entities = llama_31_infer(current_paper) # 在函数内把current_paper给改了，把title加到了abstract上
        # pred_gpt4o_entities = gpt_4o_infer(current_paper) # 在函数内把current_paper给改了，把title加到了abstract上
        
        # -- pred：两模型生成bio list -- #
        pred_llama_bio_list = convert_txt_to_bio(current_paper["abstract"], pred_llama_entities["software"])
        # pred_gpt4o_bio_list = convert_txt_to_bio(current_paper["abstract"], pred_gpt4o_entities["software"])
        llama_all_pred += pred_llama_bio_list
        
        # gpt4o_all_pred += pred_gpt4o_bio_list
        
        #  -- gold：ann的bio list -- #
        all_gold += gold_current_label
        

        assert(len(pred_llama_bio_list) == len(gold_current_label))
#%%
    print("Lenient metric")
    llama_metrics = classification_report(tags_true=all_gold, tags_pred=llama_all_pred, mode="lenient", verbose=True) # for lenient match
    print("The result of LLaMA3.1 is", dict(llama_metrics))
    print("Strict metric")
    llama_metrics = classification_report(tags_true=all_gold, tags_pred=llama_all_pred, mode="strict", verbose=True)
    print("The result of LLaMA3.1 is", dict(llama_metrics))
    
    # gpt4o_metrics = classification_report(tags_true=all_gold, tags_pred=gpt4o_all_pred, mode="lenient") # for lenient match
    # "lenient", "strict"
    # print("The result of Gpt-4o is", gpt4o_metrics)
    
        
        

# %%
