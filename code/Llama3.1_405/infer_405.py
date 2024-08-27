from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import json




# 从AutoModel中选特定的模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-70B"
)
# 从AutoTokenizer中选特定的Tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-70B")