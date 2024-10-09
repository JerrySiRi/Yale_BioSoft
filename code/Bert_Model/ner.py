from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import json
import sys
import datetime
import time
import numpy as np
import torch
import torch.nn.functional as F
import pickle
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm, trange

#from tensorboardX import SummaryWriter

from utils import NerProcessor, convert_examples_to_features, get_Dataset
# from models import BERT_BiLSTM_CRF
import conlleval as conlleval

from transformers import (WEIGHTS_NAME, BertConfig, BertTokenizer)
from transformers import AdamW, get_linear_schedule_with_warmup
from dotenv import load_dotenv




# ----------------------- 加载.env文件中的信息 ------------------- #
load_dotenv()
PYTHONPATH = os.getenv('PYTHONPATH')
print(PYTHONPATH)
sys.path.append(PYTHONPATH)
from ner_metrics_both import classification_report


logger = logging.getLogger(__name__)

# 【【CUDA Debug】】 
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'




# set the random seed for repeat
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def evaluate(args, data, model, id2label, all_ori_tokens):
    model.eval()
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=args.train_batch_size)

    logger.info("***** Running eval *****")
    # logger.info(f" Num examples = {len(data)}")
    # logger.info(f" Batch size = {args.eval_batch_size}")
    pred_labels = []
    ori_labels = []

    for b_i, (input_ids, input_mask, segment_ids, label_ids) in enumerate(tqdm(dataloader, desc="Evaluating")):
        
        input_ids = input_ids.to(args.device)
        input_mask = input_mask.to(args.device)
        segment_ids = segment_ids.to(args.device)
        label_ids = label_ids.to(args.device)

        with torch.no_grad():

            outputs = model(input_ids = input_ids)
            # 获取预测结果
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

        # 原来循环的是logits
        for l in predictions:
            pred_labels.append([id2label[int(idx)] for idx in l])
        
        for l in label_ids:
            ori_labels.append([id2label[int(idx.item())] for idx in l])
    
    eval_list = []
    for ori_tokens, oril, prel in zip(all_ori_tokens, ori_labels, pred_labels):
        for ot, ol, pl in zip(ori_tokens, oril, prel):
            if ot in ["[CLS]", "[SEP]"]:
                continue
            eval_list.append(f"{ot} {ol} {pl}\n")
        eval_list.append("\n")
    
    full_ori_labels = list()
    full_pred_labels = list()

    # 去掉末尾padding的I
    for index in range(0, len(ori_labels)):
        valid_index = len(ori_labels)
        while valid_index > 0 and ori_labels[index][valid_index - 1] == "I":
            valid_index -= 1
        full_ori_labels.extend(ori_labels[index][:valid_index])
        full_pred_labels.extend(pred_labels[index][:valid_index])
    print(full_ori_labels[:256])
    print(full_pred_labels[:256])
    print("Lenient")
    lenient_metrics = classification_report(tags_true=full_ori_labels, tags_pred=full_pred_labels, mode="lenient") # for lenient match
    print(lenient_metrics)
    print("Strict")
    strict_metrics = classification_report(tags_true=full_ori_labels, tags_pred=full_pred_labels, mode="strict") # for strict match
    print(strict_metrics)

    # eval the model 
    ###counts = conlleval.evaluate(eval_list)
    ###conlleval.report(counts)

    # namedtuple('Metrics', 'tp fp fn prec rec fscore')
    ###overall, by_type = conlleval.metrics(counts)    
    ###return overall, by_type
    print("&"*10,dict(lenient_metrics["default"]).keys)
    return dict(lenient_metrics["default"]), list()





def match_dim_tokenizer_model(tokenizer, model, device):
    # tokenizer的词汇表大小
        vocab_size = len(tokenizer)

        # 获取当前的词嵌入层
        original_embeddings = model.get_input_embeddings()

        # 获取当前的词嵌入层的大小
        original_vocab_size, embedding_dim = original_embeddings.weight.size()

        # 如果分词器的词汇表大小与模型的词嵌入层大小不匹配，调整词嵌入层
        if vocab_size != original_vocab_size:
            print(f"Adjusting embedding layer from size {original_vocab_size} to {vocab_size}")
            new_embeddings = torch.nn.Embedding(vocab_size, embedding_dim)

        # 复制原有的词嵌入矩阵到新的词嵌入矩阵
        if vocab_size < original_vocab_size: # tokenizer的词矩阵更小
            with torch.no_grad():
                new_embeddings.weight[:vocab_size, :] = original_embeddings.weight[:vocab_size, :]
            model.set_input_embeddings(new_embeddings)
        else:   # model的embedding层更小
            with torch.no_grad():
                new_embeddings.weight[:original_vocab_size, :] = original_embeddings.weight[:original_vocab_size, :]
            model.set_input_embeddings(new_embeddings)

        # 调整模型的词嵌入层
        model.resize_token_embeddings(vocab_size)

        model.to(device)
        # tokenizer.to(device)



def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_file", default=None, type=str)
    parser.add_argument("--eval_file", default=None, type=str)
    parser.add_argument("--test_file", default=None, type=str)
    parser.add_argument("--model_name_or_path", default=None, type=str)
    parser.add_argument("--output_dir", default="../output", type=str)

    ## other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--do_train", default=False, type=boolean_string)
    parser.add_argument("--do_eval", default=False, type=boolean_string)
    parser.add_argument("--do_test", default=False, type=boolean_string)
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--eval_batch_size", default=8, type=int)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--num_train_epochs", default=10, type=float)
    parser.add_argument("--warmup_proprotion", default=0.1, type=float)
    parser.add_argument("--use_weight", default=1, type=int)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--fp16", default=False)
    parser.add_argument("--loss_scale", type=float, default=0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--do_lower_case", action='store_true')
    parser.add_argument("--logging_steps", default=500, type=int)
    parser.add_argument("--clean", default=False, type=boolean_string, help="clean the output dir")

    parser.add_argument("--need_birnn", default=False, type=boolean_string)
    parser.add_argument("--rnn_dim", default=128, type=int)

    

    args = parser.parse_args()

    device = torch.device("cuda")
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_
    args.device = device
    n_gpu = torch.cuda.device_count()

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)

    logger.info(f"device: {device} n_gpu: {n_gpu}")

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    # now_time = datetime.datetime.now().strftime('%Y-%m-%d_%H')
    # tmp_dir = args.output_dir + '/' +str(now_time) + '_ernie'
    # if not os.path.exists(tmp_dir):
    #     os.makedirs(tmp_dir)
    # args.output_dir = tmp_dir
    if args.clean and args.do_train:
        # logger.info("清理")
        if os.path.exists(args.output_dir):
            def del_file(path):
                ls = os.listdir(path)
                for i in ls:
                    c_path = os.path.join(path, i)
                    print(c_path)
                    if os.path.isdir(c_path):
                        del_file(c_path)
                        os.rmdir(c_path)
                    else:
                        os.remove(c_path)
            try:
                del_file(args.output_dir)
            except Exception as e:
                print(e)
                print('pleace remove the files of output dir and data.conf')
                exit(-1)
    
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.exists(os.path.join(args.output_dir, "eval")):
        os.makedirs(os.path.join(args.output_dir, "eval"))
    
    # writer = SummaryWriter(logdir=os.path.join(args.output_dir, "eval"), comment="Linear")

    processor = NerProcessor()
    label_list = processor.get_labels(args)
    num_labels = len(label_list)
    args.label_list = label_list

    if os.path.exists(os.path.join(args.output_dir, "label2id.pkl")):
        with open(os.path.join(args.output_dir, "label2id.pkl"), "rb") as f:
            label2id = pickle.load(f)
    else:
        label2id = {l:i for i,l in enumerate(label_list)}
        with open(os.path.join(args.output_dir, "label2id.pkl"), "wb") as f:
            pickle.dump(label2id, f)      
    
    id2label = {value:key for key,value in label2id.items()} 

    # Prepare optimizer and schedule (linear warmup and decay)

# TODO:【【【【【【【【【【【【【【【更改！】】】】】】】】】】】】】】】】】】】】】】】
# tokenizer、model、数据集获取方式 & 处理方式
# 问chatgpt：本地数据集分为data和label，如何用hugging face 模型的api进行处理

    from transformers import AutoTokenizer, AutoModelForTokenClassification, \
                            TrainingArguments, Trainer, AutoConfig

    if args.do_train:

        """
        tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, 
                    do_lower_case=args.do_lower_case)
        config = BertConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, 
                num_labels=num_labels)
        model = BERT_BiLSTM_CRF.from_pretrained(args.model_name_or_path, config=config, 
                need_birnn=args.need_birnn, rnn_dim=args.rnn_dim)
        
        """
        tokenizer = AutoTokenizer.from_pretrained("Clinical-AI-Apollo/Medical-NER",
                    truncation=True, 
                    padding=True, 
                    is_split_into_words=True, 
                    return_tensors="pt")

        # preassigned hyper-parameter
        config = AutoConfig.from_pretrained(
            "Clinical-AI-Apollo/Medical-NER", 
            # hidden_size=768,       
            # num_attention_heads=12, 
            # num_hidden_layers=12,   
            # intermediate_size=3072, 
            hidden_dropout_prob=0.1, # hidden layer dropout pro
            attention_probs_dropout_prob=0.1, # attention dropout pro
            # max_position_embeddings=512,      
            # type_vocab_size=2,        # 词汇类型嵌入大小 (用于 token 类型嵌入)
            # initializer_range=0.02,   # 参数初始化范围
            layer_norm_eps=1e-12,     # 层归一化 epsilon 值
            output_attentions=False,  # 是否输出注意力权重
            output_hidden_states=False, # 是否输出隐藏层状态
            num_labels = 3,             # 分类任务中的标签数
            id2label={0: 'O', 1: 'B', 2: 'I'},  
            label2id={'O': 0, 'B': 1, 'I': 2}, 
        )

        # Use predefined parameters 
        model = AutoModelForTokenClassification.from_pretrained("Clinical-AI-Apollo/Medical-NER", \
                                                                config=config, ignore_mismatched_sizes = True)

        """
        model.to(device)
        # 获取分词器的词汇表大小
        vocab_size = len(tokenizer)
        print(f"Vocabulary size: {vocab_size}")

        # 获取模型的嵌入层大小
        embedding_size = model.get_input_embeddings().weight.size(0)
        print(f"Embedding size: {embedding_size}")

        # 确保词汇表大小与嵌入层大小一致
        assert vocab_size == embedding_size, "Vocabulary size and embedding size do not match"
        """

        match_dim_tokenizer_model(tokenizer, model, device)


        if n_gpu > 1: #【使用多个gpu并行训练！】
            model = torch.nn.DataParallel(model)

        train_examples, train_features, train_data = get_Dataset(args, processor, tokenizer, mode="train")
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        match_dim_tokenizer_model(tokenizer, model, device)


        if args.do_eval:
            eval_examples, eval_features, eval_data = get_Dataset(args, processor, tokenizer, mode="eval")
            match_dim_tokenizer_model(tokenizer, model, device)
      
        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0.1*t_total,
            num_training_steps=t_total
        )


        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_data))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Total optimization steps = %d", t_total)

        


        model.train()
        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        best_f1 = 0.0
        for ep in trange(int(args.num_train_epochs), desc="Epoch"):
            model.train()
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                
                input_ids, input_mask, segment_ids, label_ids = batch
                outputs = model(input_ids, labels=label_ids)
                # segment_ids, input_mask
                loss = outputs.loss

                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()
                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1
                    
                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        tr_loss_avg = (tr_loss-logging_loss)/args.logging_steps
                        # writer.add_scalar("Train/loss", tr_loss_avg, global_step)
                        logging_loss = tr_loss
            
            if args.do_eval:

                all_ori_tokens_eval = [f.ori_tokens for f in eval_features]
                overall, by_type = evaluate(args, eval_data, model, id2label, all_ori_tokens_eval)
                


                # add eval result to tensorboard
                f1_score = overall["f1-score"]
                ###writer.add_scalar("Eval/precision", overall.prec, ep)
                ###writer.add_scalar("Eval/recall", overall.rec, ep)
                ###writer.add_scalar("Eval/f1_score", overall.fscore, ep)

                # save the best performs model
                if f1_score > best_f1:
                    logger.info(f"----------the best f1 is {f1_score}---------")
                    best_f1 = f1_score
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(args.output_dir, save_config=True) # 把模型保存到指定文件夹
                    tokenizer.save_pretrained(args.output_dir)

                    # Good practice: save your training arguments together with the trained model
                    torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

            # logger.info(f'epoch {ep}, train loss: {tr_loss}')
        # writer.add_graph(model)
        # writer.close()

        # model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        # model_to_save.save_pretrained(args.output_dir)
        # tokenizer.save_pretrained(args.output_dir)

        # TODO：Good practice: save your training arguments together with the trained model
        # torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))


    if args.do_test:
        # model = BertForTokenClassification.from_pretrained(args.output_dir)
        # model.to(device)
        # tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        # args = torch.load(os.path.join(args.output_dir, 'training_args.bin'))
        # model = BERT_BiLSTM_CRF.from_pretrained(args.output_dir, need_birnn=args.need_birnn, rnn_dim=args.rnn_dim)


        label_map = {i : label for i, label in enumerate(label_list)}

        tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
        model = AutoModelForTokenClassification.from_pretrained(args.output_dir)
        model.to(device)

        test_examples, test_features, test_data = get_Dataset(args, processor, tokenizer, mode="test")

        logger.info("***** Running test *****")
        logger.info(f" Num examples = {len(test_examples)}")
        logger.info(f" Batch size = {args.eval_batch_size}")

        all_ori_tokens = [f.ori_tokens for f in test_features]
        all_ori_labels = [e.label.split(" ") for e in test_examples]
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)
        model.eval()

        pred_labels = []
        
        for b_i, (input_ids, input_mask, segment_ids, label_ids) in enumerate(tqdm(test_dataloader, desc="Predicting")):
            
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                logits = model.predict(input_ids, segment_ids, input_mask)
            # logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
            # logits = logits.detach().cpu().numpy()

            for l in logits:

                pred_label = []
                for idx in l:
                    pred_label.append(id2label[idx])
                pred_labels.append(pred_label)

        assert len(pred_labels) == len(all_ori_tokens) == len(all_ori_labels)
        print(len(pred_labels))
        with open(os.path.join(args.output_dir, "token_labels_.txt"), "w", encoding="utf-8") as f:
            for ori_tokens, ori_labels,prel in zip(all_ori_tokens, all_ori_labels, pred_labels):
                for ot,ol,pl in zip(ori_tokens, ori_labels, prel):
                    if ot in ["[CLS]", "[SEP]"]:
                        continue
                    else:
                        f.write(f"{ot} {ol} {pl}\n")
                f.write("\n")

if __name__ == "__main__":
    main()
    pass