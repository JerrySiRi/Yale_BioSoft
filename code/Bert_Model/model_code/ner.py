from __future__ import absolute_import, division, print_function
import argparse
import logging
import os
import random
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
import pickle
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                             TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
#from tensorboardX import SummaryWriter

from transformers import (WEIGHTS_NAME, BertConfig, BertTokenizer)
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForTokenClassification, \
                            TrainingArguments, Trainer, AutoConfig
from dotenv import load_dotenv

from ner_metrics_both import classification_report


# ----------------------- 加载.env文件中的信息 ------------------- #
load_dotenv()
PYTHONPATH = os.getenv('PYTHONPATH')
print(PYTHONPATH)
sys.path.append(PYTHONPATH)
logger = logging.getLogger(__name__)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from Bert_Model.model_code.utils import NerProcessor, convert_examples_to_features, get_Dataset
import Bert_Model.model_code.conlleval as conlleval


# --- set the random seed for repeat --- #
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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
            outputs = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, labels=label_ids)
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
    

    print("--- Pred Labels --- \n")
    print(pred_labels)

    plain_ori_labels = list()
    plain_pred_labels = list()
    for ori, pred in zip(ori_labels, pred_labels):
        plain_ori_labels.extend(ori)
        plain_pred_labels.extend(pred)


    print("\n --- Lenient --- \n")
    lenient_metrics = classification_report(tags_true=plain_ori_labels, tags_pred=plain_pred_labels, mode="lenient") # for lenient match
    print(lenient_metrics)
    print("\n --- Strict --- \n")
    strict_metrics = classification_report(tags_true=plain_ori_labels, tags_pred=plain_pred_labels, mode="strict") # for strict match
    print(strict_metrics)

    # eval the model 
    ### counts = conlleval.evaluate(eval_list)
    ### conlleval.report(counts)

    # namedtuple('Metrics', 'tp fp fn prec rec fscore')
    ### overall, by_type = conlleval.metrics(counts)    
    ### return overall, by_type

    print("&"*10, dict(lenient_metrics['']).keys)
    return dict(lenient_metrics['']), list()



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



def main(args=None, infernece_file=None, pipeline=False):
    
    if pipeline == False:
        parser = argparse.ArgumentParser(
            description='Extract software names from academic papers',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            # 帮助格式化器，它会显示命令行参数的默认值。
        )

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
        

        parser.add_argument("--max_seq_length", default=512, type=int)
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
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--fp16", default=False)
        parser.add_argument("--loss_scale", type=float, default=0)
        parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
        parser.add_argument("--warmup_steps", default=0, type=int)
        parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        parser.add_argument("--max_steps", default=-1, type=int)
        parser.add_argument("--do_lower_case", action='store_true')
        parser.add_argument("--logging_steps", default=500, type=int)
        parser.add_argument("--clean", default=False, type=boolean_string, help="clean the output dir")
        parser.add_argument("--push_hf", default=False, type=boolean_string, help="clean the output dir")


        parser.add_argument("--need_birnn", default=False, type=boolean_string)
        parser.add_argument("--rnn_dim", default=128, type=int)

        
        # --- test时使用 --- #
        parser.add_argument('--action', type=str, help='The action to perform: demo, extract', choices=['demo', 'extract'])
        # 可选参数，使用--之后才给，也可以不给
        parser.add_argument('--each_year_sample_size', type=int, default=10, help='The number of samples to extract')
        parser.add_argument('--pmid_filter', type=str, default='1=1', help='Other specific filter instrction')
        # 永真式，不需要就没用，需要再改
        parser.add_argument('--start_year', type=str, default='2009', help='The start year of papers to extract')
        parser.add_argument('--end_year', type=str, default='2023', help='The end year of papers to extract')
        parser.add_argument('--percent', type=float, default=1, help='In-context shots. Choice: 4, 8, 16')
        args = parser.parse_args()
    else:
        args = args

    device = torch.device("cuda")
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_
    args.device = device
    n_gpu = torch.cuda.device_count()
    set_seed(args)

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
        logger.info("clean = True, clean the saved model")
        if os.path.exists(args.output_dir):
            def del_file(path):
                ls = os.listdir(path)
                for i in ls:
                    c_path = os.path.join(path, i)
                    print(c_path)
                    if os.path.isdir(c_path):
                        del_file(c_path)
                    else:
                        os.remove(c_path)
            try:
                del_file(args.output_dir)
            except Exception as e:
                print(e)
                print('pleace remove the files of output dir and data.conf')
                exit(-1)
    
    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        # raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    

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


    # ----------- Training Phase ----------- #

    if args.do_train:
        # use huggingface model, which is (pre) fine-tuned on Medical Entities with NER tasks
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base",
                    truncation=True, 
                    padding=True, 
                    is_split_into_words=True, 
                    return_tensors="pt")

        # preassigned hyper-parameter
        config = AutoConfig.from_pretrained(
            "microsoft/deberta-v3-base", 
            # hidden_size=768,       
            # num_attention_heads=12, 
            # num_hidden_layers=12,   
            # intermediate_size=3072, 
            # hidden_dropout_prob=0.1, # hidden layer dropout pro
            # attention_probs_dropout_prob=0.1, # attention dropout pro
            # max_position_embeddings=512,      
            # type_vocab_size=2,        # 词汇类型嵌入大小 (用于 token 类型嵌入)
            # initializer_range=0.02,   # 参数初始化范围
            # layer_norm_eps=1e-12,     # 层归一化 epsilon 值
            output_attentions=False,  # 是否输出注意力权重
            output_hidden_states=False, # 是否输出隐藏层状态
            num_labels = 3,             
            id2label={0: 'O', 1: 'B', 2: 'I'},  
            label2id={'O': 0, 'B': 1, 'I': 2}, 
        )

        # Use predefined parameters 
        model = AutoModelForTokenClassification.from_pretrained("microsoft/deberta-v3-base", \
                                                                config = config, \
                                                                ignore_mismatched_sizes = True)
        
        # ----- fix bug with unmatched dimension between tokenizer and model ----- #
        match_dim_tokenizer_model(tokenizer, model, device)

        if n_gpu > 1: #【使用多个gpu并行训练！】
            model = torch.nn.DataParallel(model)
        

        # ----- Data & Optimizer prepare ----- #

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


        # --- Train --- #
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
                outputs = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, labels=label_ids)
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
                

                # add eval result to tensorboard/txt
                f1_score = overall["f1-score"]
                with open(os.path.join(str(args.output_dir)+"/train", "training_performance.txt"), "a", encoding="utf-8") as f:
                        p = overall["precision"]
                        r = overall["recall"]
                        f1 = overall["f1-score"]
                        f.write(f"""
                                Batch {ep} Lenient Performace:
                                \tprecision: {p}
                                \trecall: {r}
                                \tf1-score: {f1}\n""")
                        f.write("\n")

                ### writer.add_scalar("Eval/precision", overall.prec, ep)
                ### writer.add_scalar("Eval/recall", overall.rec, ep)
                ### writer.add_scalar("Eval/f1_score", overall.fscore, ep)

                # save the best performs model
                if f1_score > best_f1:
                    logger.info(f"\n----------the best f1 is ---------\n{f1_score}")
                    best_f1 = f1_score

                    assert model.config.vocab_size == len(tokenizer)

                    if args.push_hf == True:
                        
                        # --- save in disk --- #
                        save_path = str(args.output_dir)+"/model"
                        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(save_path, save_config=True) # 把模型保存到指定文件夹
                        tokenizer.save_pretrained(save_path)

                        # --- save in huggingface --- #
                        model.push_to_hub("JerrySiRi/SWN_Biomedical_NER_Bert", tokenizer, save_method = "merged_16bit", token = "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")
                        tokenizer.push_to_hub("JerrySiRi/SWN_Biomedical_NER_Bert", token = "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")

                        # Good practice: save your training arguments together with the trained model
                        torch.save(args, os.path.join(save_path, 'training_args.bin'))

            logger.info(f'epoch {ep}, train loss: {tr_loss}')
        # writer.add_graph(model)
        # writer.close()




    # ----------- Testing Phase ----------- #

    if args.do_test:
        # ../../../datasets/PubMed + f"/Pubmed_{START_YEAR}_{END_YEAR}_{SAMPLE_SIZE}.txt"
        # args.test_file = "../../../datasets/PubMed/Pubmed_2023_2023_100.txt"
        if pipeline == True:
            args.test_file = infernece_file

        def model_tokenizer_test(input_ids, tokenizer, model):
            if (input_ids.max() > model.config.vocab_size) or \
                (label_ids.max() > model.config.label2id) or \
                (tokenizer.vocab_size != model.config.vocab_size):
                print("Unmatched tokenizer and model embedding dimension")
                return False
            else:
                return True

        tokenizer = AutoTokenizer.from_pretrained("JerrySiRi/SWN_Biomedical_NER_Bert",
                    truncation=True, 
                    padding=True, 
                    is_split_into_words=True, 
                    return_tensors="pt")
        config = AutoConfig.from_pretrained(
            "JerrySiRi/SWN_Biomedical_NER_Bert", 
            output_attentions=False,  # 是否输出注意力权重
            output_hidden_states=False, # 是否输出隐藏层状态
            num_labels = 3,             
            id2label={0: 'O', 1: 'B', 2: 'I'},  
            label2id={'O': 0, 'B': 1, 'I': 2}, 
        )
        model = AutoModelForTokenClassification.from_pretrained("JerrySiRi/SWN_Biomedical_NER_Bert", \
                                                                config = config, \
                                                                ignore_mismatched_sizes = True)
    
        model.to(device)
        
        max_length = tokenizer.model_max_length
        print(f"--- Tokenizer max length: {max_length} ---")
        print(f"--- Model max length: {model.config.max_position_embeddings} ---")

        # 前两个是列表的列表，test_data是值
        test_label_examples, test_features, test_data = get_Dataset(args, processor, tokenizer, mode="test")

        logger.info("***** Running test *****")
        logger.info(f" Num examples = {len(test_label_examples)}")
        logger.info(f" Batch size = {args.eval_batch_size}")

        # assert len(pred_labels) == len(all_ori_tokens) == len(all_ori_labels)。第三个比前两个少了3个

        all_ori_tokens = [f.ori_tokens for f in test_features]
        all_ori_labels = test_label_examples

        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)
        model.eval()


        pred_labels = []  
        # if model_tokenizer_test(input_ids, tokenizer, model) == False: 

        # BUG BUG BUG：重新给他了一个新的embedding层，没有用原始fine-tune过的！！！ 
        match_dim_tokenizer_model(tokenizer, model, device)
        

        # ----- Inference:  Use token_ids ----- #
        # CRF和Bert结合在一起了，此时model的输出结果是一个个的tensor，取max的就行

        for b_i, (input_ids, input_mask, segment_ids, label_ids) in enumerate(tqdm(test_dataloader, desc="Predicting")):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            # test_dataloader是batch化过的啦
            with torch.no_grad():
                # print(input_ids[0], segment_ids[0], input_mask[0])
                outputs = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)

            # -- Let argmax label represent its tag -- #
            predictions = torch.argmax(F.log_softmax(outputs.logits, dim=-1), dim=-1)
            # -- Convert tensor value into numpy value for indexing -- #
            predictions = predictions.detach().cpu().numpy()
            # print("---Predictions---\n", predictions)
            
            for l in predictions:
                pred_label = []
                for idx in l:
                    pred_label.append(id2label[idx])
                pred_labels.append(pred_label)

        assert len(pred_labels) == len(all_ori_tokens) == len(all_ori_labels),\
            f"pred:{len(pred_labels)}, ori_token:{len(all_ori_tokens)}, ori_labels:{len(all_ori_labels)}"

        print(f"\n --- Length of *chunked* prediction labels: {len(pred_labels)} --- \n")

        # prel和ori_token没错位！正常解析就好！

        target_file = ""
        if pipeline == True:
            target_file = "PubMed_labels.txt"
        else:
            target_file = "test_labels.txt"

        with open(os.path.join(str(args.output_dir)+"/inference", target_file), "w", encoding="utf-8") as f:
            for _, (ori_tokens, ori_labels, prel) in enumerate(zip(all_ori_tokens, all_ori_labels, pred_labels)):
                # mismatch of the inference --- assign tag on special token
                for ot, ol, pl in zip(ori_tokens, ori_labels, prel):
                    # print("--- ot, ol, pl ---\n")
                    # print(ot, ol, pl)
                    if ot in ["[CLS]", "[SEP]"]:
                        continue
                    else:
                        f.write(f"{ot}\t{ol}\t{pl}\n")
                f.write("\n")



if __name__ == "__main__":
    main()
    pass