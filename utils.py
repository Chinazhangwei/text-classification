import numpy as np
from tqdm import tqdm
from config import Config
import pandas as pd
from transformers import BertTokenizer


def load_data(data_file):
    """
    读取数据
    :param file:
    :return:
    """
    data_df = pd.read_csv(data_file)
    lines = list(zip(list(data_df['sentence']), list(data_df['num_label'])))
    return lines


def create_example(lines):
    examples = []
    for (i, line) in enumerate(lines):
        sentence = line[0]
        label = int(line[1])
        examples.append(InputExample(sentence=sentence, label=label))
    return examples[:]


def get_examples(data_file):
    return create_example(
        load_data(data_file)
    )


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, sentence, label=None):
        self.sentence = sentence
        self.label = label


class DataIterator:
    """
    数据迭代器
    """
    # batch_size 是config里面的一些值 data_file就是csv文件
    def __init__(self, batch_size, data_file, tokenizer, use_bert=False, seq_length=100, is_test=False):
        self.data_file = data_file
        self.data = get_examples(data_file)
        self.batch_size = batch_size
        self.use_bert = use_bert
        self.seq_length = seq_length
        self.num_records = len(self.data) # 数据数量
        self.all_tags = [] # 序号
        self.idx = 0  # 数据索引
        self.all_idx = list(range(self.num_records))  # 全体数据索引
        self.is_test = is_test # is_test表示是否是验证集或者预测集，需要shuffle(),不是的话是训练集，训练集需要shuffle()打乱数据增强一下
        if not self.is_test:
            self.shuffle() # all_idx会被打乱
        self.tokenizer = tokenizer # bert自带切齿器
        print(self.num_records)
    # 构造数据去喂 输出res
    def convert_single_example(self, example_idx):
        sentence = self.data[example_idx].sentence

        label = self.data[example_idx].label
        """得到input的token-----start-------"""
        q_tokens = []
        ntokens = []
        segment_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        # 得到text_a的token
        for word in sentence:
            token = self.tokenizer.tokenize(word)
            q_tokens.extend(token)
        # 把token加入至所有字的token中

        for token in q_tokens:
            ntokens.append(token)
            segment_ids.append(0)
        ntokens.append("[SEP]")
        segment_ids.append(1)

        ntokens = ntokens[:self.seq_length - 1]
        segment_ids = segment_ids[:self.seq_length - 1]

        ntokens.append("[SEP]")
        segment_ids.append(1)
        """得到input的token-------end--------"""

        """token2id---start---"""
        input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)

        # padding补0
        while len(input_ids) < self.seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            # we don't concerned about it!
            ntokens.append("**NULL**")

        assert len(input_ids) == self.seq_length
        assert len(input_mask) == self.seq_length
        assert len(segment_ids) == self.seq_length
        """token2id ---end---"""
        return input_ids, input_mask, segment_ids, label

    def shuffle(self):
        np.random.shuffle(self.all_idx)

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= self.num_records:  # 迭代停止条件 大于数据样本
            self.idx = 0
            if not self.is_test:
                self.shuffle()
            raise StopIteration

        input_ids_list = []
        input_mask_list = []
        segment_ids_list = []
        label_list = []
        num_tags = 0
        while num_tags < self.batch_size:  # 每次返回batch_size个数据
            idx = self.all_idx[self.idx]
            res = self.convert_single_example(idx)
            if res is None:
                self.idx += 1
                if self.idx >= self.num_records:
                    break
                continue
            input_ids, input_mask, segment_ids, labels = res

            # 一个batch的输入
            input_ids_list.append(input_ids)
            input_mask_list.append(input_mask)
            segment_ids_list.append(segment_ids)
            label_list.append(labels)
            if self.use_bert:
                num_tags += 1

            self.idx += 1
            if self.idx >= self.num_records:
                break
        while len(input_ids_list) < self.batch_size:
            input_ids_list.append(input_ids_list[0])
            input_mask_list.append(input_mask_list[0])
            segment_ids_list.append(segment_ids_list[0])
            label_list.append(label_list[0])
        # 传到模型输入里面
        return input_ids_list, input_mask_list, segment_ids_list, label_list, self.seq_length


if __name__ == '__main__':
    config = Config()
    # 读取数据迭代器
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=config.model_path,
                                              do_lower_case=True,
                                              never_split=["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"])
    train_iter = DataIterator(config.batch_size,
                              data_file=config.base_dir + 'train.csv', use_bert=config.use_bert,
                              tokenizer=tokenizer, seq_length=config.sequence_length)
    dev_iter = DataIterator(config.batch_size,
                            data_file=config.base_dir + 'dev.csv', use_bert=config.use_bert,
                            seq_length=config.sequence_length, is_test=True, tokenizer=tokenizer)
    for input_ids_list, input_mask_list, segment_ids_list, labels_list, seq_length in tqdm(dev_iter):
        print(input_ids_list[-1])
        print(labels_list[-1])

        break
