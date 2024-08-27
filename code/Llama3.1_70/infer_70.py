# 管道中自定义预训练模型与分词器
#%%
from transformers import pipeline, AutoConfig, AutoModelForCausalLM, AutoTokenizer
import json
import os
from huggingface_hub import login







# ----------------------- 加载.env文件中的信息 ------------------- #

HF_TOKEN = "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"
# 使用你的访问令牌进行身份验证
login(token=HF_TOKEN)

# 加载配置并设置自定义的 rope_scaling
"""
config = AutoConfig.from_pretrained("meta-llama/Meta-Llama-3.1-70B-Instruct")
config.rope_scaling = {
    "type": "linear",  # 假设使用线性缩放
    "factor": 8.0
}
"""


# 加载模型和tokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-70B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-70B-Instruct")

# 初始化pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)



TPL_PROMPT = """

Title: {title}
     
Abstract: {abstract}
"""
# 这个会比annotator要好！
SYSTEM_ROLE = "You are an experienced software developer, data scientist, and researcher in biomedical fields, skilled in developing software using various techniques and particularly well-versed in the names of software used in this domain."

#%%

def infer_70(system_role, prompt_template, paper, prompt_number):
    guidelines = """"""
    few_shots = """"""
    with open("../../datasets/prompts/guidelines.txt", 'r', encoding='utf-8') as file_txt:
        for line in file_txt:
            guidelines += line
    with open(f"../../datasets/prompts/few_shots_Llama31_{prompt_number}.txt", 'r', encoding='utf-8') as file_txt:
        for line in file_txt:
            few_shots += line
    try:
        # 传入的是TPL_prompt, 里边有format函数要用的{title}和{abstract}。
        # 传入的paper会给键值对
        # 【改】把原来的prompt加到了abstract里边
        paper["abstract"] = paper["title"] + paper["abstract"]
        prompt = prompt_template.format(**paper)

        # 返回的是一个json对象，有"software"关键字

        Prompt_all = f"""# You are given a title and an abstract of an academic publication. Your task is to identify and extract the names of software mentioned in the abstract. Software names are typically proper nouns and may include specific tools, platforms, or libraries used in research. Please list the software names you find in the publication in a JSON object using a key "software". If you are unable to identify any software names, please return an empty list of "software". When identifying software names, please consider the following exclusion criteria 
                            Also, apply following \"Guidelines\" and refer following \"Gold Examples\" to help with accuracy \n"""\
                            f"# Guidelines: {guidelines} \n"\
                            f"# Gold Examples: {few_shots} \n"\
                            f"# INPUT: {prompt} \n"\
                            f"\n"\
                            f"# OUTPUT: \n"
        
        
        messages = [
            {"role": "system", "content":system_role},
            {"role": "user", "content": Prompt_all},
        ]
        
        completion = pipe(messages)

        results_json = json.dumps(completion, indent=4)

        return results_json
    
    except Exception as e:
        print(f'! error: {e}')
        # print full stack
        import traceback
        traceback.print_exc()

        return None


if __name__ == "__main__":
    demo_paper = """
    IBIS integrated biological imaging system : electron micrograph image - processing software running on Unix workstations .
' IBIS ' is a set of computer programs concerned with the processing of electron micrographs , with particular emphasis on the requirements for structural analyses of biological macromolecules .
The software is written in FORTRAN 77 and runs on Unix workstations .
A description of the various functions and the implementation mode is given .
Some examples illustrate the user interface .
    """
    result = infer_70(TPL_PROMPT, demo_paper, 16)







# %%
