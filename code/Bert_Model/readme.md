
# 项目架构

.
├── data_postprocess -- 数据后处理，txt（或一开始就用db）转化成db格式
│   ├── db.py
│   ├── llama_infer.py
│   └── notebook.py
├── data_preprocess -- 数据预处理：train, dev, test的划分 & PubMed数据读取转化成test的格式
│   └── data_preprocess.py
├── Instruction_Bert.txt
├── model_code
│   ├── clue_process.py
│   ├── conlleval.py
│   ├── models.py
│   ├── ner.py
│   ├── __pycache__
│   └── utils.py
├── output
│   ├── eval
│   ├── inference
│   ├── label2id.pkl
│   ├── label_list.pkl
│   └── model
├── __pycache__
│   ├── conlleval.cpython-311.pyc
│   ├── conlleval.cpython-312.pyc
│   ├── conlleval.cpython-39.pyc
│   ├── utils.cpython-311.pyc
│   ├── utils.cpython-312.pyc
│   └── utils.cpython-39.pyc
├── readme.md
└── structure.txt

10 directories, 20 files
