import logging
import os
import sys
import torch
import pickle

from torch.utils.data import TensorDataset
from tqdm import tqdm



"""


"""


logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for simple sequence classification.
        
        Function: single input class
    """
    

    def __init__(self, guid, text, label=None):
        self.guid = guid
        self.text = text 
        self.label = label


class InputFeatures(object):
    """A single set of features of data.
    
        Function； single input feature
    """

    def __init__(self, input_ids, input_mask, segment_ids, label_id, ori_tokens):

        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.ori_tokens = ori_tokens



class NerProcessor(object):
    def read_data(self, input_file):

        """Reads a BIO data.
        
            input_file: BIO annotated data
        """

        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            words = [] # word
            labels = [] # B/I/O label
            
            for line in f.readlines():   
                contends = line.strip()
                tokens = line.strip().split("\t") # "token  BIO type" 
                if len(tokens) == 2: # have content
                    words.append(tokens[0])
                    labels.append(tokens[1])
                else: # finish process
                    if len(contends) == 0 and len(words) > 0:
                        label = []
                        word = []
                        for l, w in zip(labels, words):
                            if len(l) > 0 and len(w) > 0:
                                label.append(l)
                                word.append(w)
                        lines.append([' '.join(label), ' '.join(word)])
                        words = []
                        labels = []
            
            return lines # ["initial text", "label of initial text"]
    
    
    def get_labels(self, args):
        labels = set()

        if os.path.exists(os.path.join(args.output_dir, "label_list.pkl")):
            logger.info(f"loading labels info from {args.output_dir}")
            with open(os.path.join(args.output_dir, "label_list.pkl"), "rb") as f:
                labels = pickle.load(f)
        else:
            # get labels from train data
            logger.info(f"loading labels info from train file and dump in {args.output_dir}")
            with open(args.train_file) as f:
                for line in f.readlines():
                    tokens = line.strip().split("\t")

                    if len(tokens) == 2:
                        labels.add(tokens[1])
            if len(labels) > 0:
                with open(os.path.join(args.output_dir, "label_list.pkl"), "wb") as f:
                    pickle.dump(labels, f)
            else:
                logger.info("loading error and return the default labels B,I,O")
                labels = {"O", "B", "I"}
        return labels 

    def get_examples(self, input_file):
        examples = []
        
        lines = self.read_data(input_file)

        for i, line in enumerate(lines):
            guid = str(i)
            text = line[1]
            label = line[0]
            examples.append(InputExample(guid=guid, text=text, label=label))

        return examples



def convert_examples_to_features(args, examples, label_list, max_seq_length, tokenizer):

    # 长度是max_length的列表
    def chunk_long_text(examples, max_length):
        examples_chunks = [examples[i: i + max_length] for i in range(0, len(examples), max_length)]
        return examples_chunks

    label_map = {label : i for i, label in enumerate(label_list)}
    features = [] 
    chunked_label_examples = []

    for (ex_index, example) in tqdm(enumerate(examples), desc="convert examples"):
        # if ex_index % 10000 == 0:
        #     logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        textlist = example.text.split(" ")
        labellist = example.label.split(" ")

        # TODO：分成多个不超过max_length的chunks
        textlist_chunks = chunk_long_text(textlist, max_seq_length)
        labellist_chunks = chunk_long_text(labellist, max_seq_length)

        for item in labellist_chunks:
            chunked_label_examples.append(item)

        # PubMed: 此时labellist全是'O', 并无实际作用
        for cur_textlist, cur_labellist in zip(textlist_chunks, labellist_chunks):
            # -- 对每一个子列表操作 -- #
            assert len(cur_textlist) == len(cur_labellist)
            tokens = [] # 每一个chunk中，tokenizer做tokenize后的结果
            labels = []
            ori_tokens = []

            # --- 把所有出现的word加到tokenizer的词表中 --- #
            for i, word in enumerate(cur_textlist):
                if len(tokenizer.tokenize(word)) != 1:
                    tokenizer.add_tokens(word) 


            for i, word in enumerate(cur_textlist):
                # 防止wordPiece情况出现，不过貌似不会
                token = tokenizer.tokenize(word)
                tokens.extend(token)
                label_1 = cur_labellist[i]
                ori_tokens.append(word)

                # 单个字符不会出现wordPiece
                for m in range(len(token)):
                    if m == 0:
                        labels.append(label_1)
                    else:
                        if label_1 == "O":
                            labels.append("O")
                        else:
                            labels.append("I")
                
            if len(tokens) >= max_seq_length - 1:
                tokens = tokens[0: (max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
                labels = labels[0: (max_seq_length - 2)]
                ori_tokens = ori_tokens[0: (max_seq_length - 2)]

            ori_tokens = ["[CLS]"] + ori_tokens + ["[SEP]"]
            
            ntokens = []
            segment_ids = []
            label_ids = []

            ntokens.append("[CLS]")
            segment_ids.append(0)
            label_ids.append(label_map["O"])
            for i, token in enumerate(tokens):
                ntokens.append(token)
                segment_ids.append(0)
                label_ids.append(label_map[labels[i]])
                
            ntokens.append("[SEP]")
            segment_ids.append(0)
            label_ids.append(label_map["O"])
            input_ids = tokenizer.convert_tokens_to_ids(ntokens)   
            
            input_mask = [1] * len(input_ids)

            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                # we don't concerned about it!
                label_ids.append(0)
                ntokens.append("**NULL**")

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length

            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("tokens: %s" % " ".join(
                    [str(x) for x in ntokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

            # 一个列表的feature
            features.append(
                    InputFeatures(input_ids=input_ids,
                                input_mask=input_mask,
                                segment_ids=segment_ids,
                                label_id=label_ids,
                                ori_tokens=ori_tokens))
 
    return features, chunked_label_examples


def get_Dataset(args, processor, tokenizer, mode="train"):
    if mode == "train":
        filepath = args.train_file
    elif mode == "eval":
        filepath = args.eval_file
    elif mode == "test":
        filepath = args.test_file
    else:
        raise ValueError("mode must be one of train, eval, or test")

    print(filepath)
    examples = processor.get_examples(filepath)
    label_list = args.label_list

    features, chunked_examples = convert_examples_to_features(
        args, examples, label_list, args.max_seq_length, tokenizer
    )
    
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)


    return chunked_examples, features, data
