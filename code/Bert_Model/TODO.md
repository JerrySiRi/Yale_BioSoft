# 1. ok-解决chunks的问题

# 2. [必须做]-解决model改embedding层无法上传的问题

# 3. ok-创建db, 如何让Bert进行inference

BUG：test数据集就可以，pubmed数据集就全部是B ->-> 数据集处理的问题？数据集来源的问题？

embedding层无法直接匹配上的问题？（感觉不像，test的时候也没匹配上，仍然可以继续做）

实验1：用现在的代码，尝试test data -> 发现出错 -> 自己的代码问题

实验2：用原来的代码，尝试pubmed data -> 运行没有错 -> 自己代码的问题

# 4. 后处理，上传到huggingface

# 5. batch操作做inference，现在是线性的（用tokenizer来做batch）

# TODO: 速度太慢了，看看能不能batch（convert的时候没有用gpu，多线程来做tokenize？）


# TODO：可以先让LLaMA来做inference，这样不用batch先把2023年跑出来
