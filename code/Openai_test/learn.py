from openai import OpenAI
import os



"""
# 【【Chat Completion -- 对话补全】】

client = OpenAI(api_key="sk-proj-OvmtynB5B3OZer0ayFJQT3BlbkFJ38o233iGXqJfwJHlDW1l")
completion = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
    {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
    {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
  ]
)
print(completion.choices[0].message)
"""



"""
# 【【embedding 生成】】

client = OpenAI(api_key="sk-proj-OvmtynB5B3OZer0ayFJQT3BlbkFJ38o233iGXqJfwJHlDW1l")

response = client.embeddings.create(
  model="text-embedding-ada-002",
  input="The food was delicious and the waiter..."
)

print(response)
"""

# 【【图像生成】】

client = OpenAI(api_key="sk-proj-OvmtynB5B3OZer0ayFJQT3BlbkFJ38o233iGXqJfwJHlDW1l")
response = client.images.generate(
  prompt="A cute baby sea otter",
  n=2,
  size="1024x1024"
)

print(response)




