import pandas as pd
from config import Config

config = Config()

train_df = pd.read_csv(config.base_dir + 'train.csv', encoding='utf8')
dev_df = pd.read_csv(config.base_dir + 'dev.csv', encoding='utf8')

def cal_text_len(row):
    row_len = len(row)
    if row_len < 256:
        return 256
    elif row_len < 384:
        return 384
    elif row_len < 512:
        return 512
    elif row_len < 1024:
        return 1024
    else:
        return 2048


train_df['text_len'] = train_df['text'].apply(cal_text_len)
dev_df['text_len'] = dev_df['text'].apply(cal_text_len)

print(train_df['text_len'].value_counts())
print(dev_df['text_len'].value_counts())

print('-------------------')


def merge_text(text):
    if len(text) < 512:
        return text
    else:
        return text[:128] + text[-382:]

#  取文本段前128与后382作为整体的文本
train_df['sentence'] = train_df['text'].apply(merge_text)
dev_df['sentence'] = dev_df['text'].apply(merge_text)


train_df['text_len'] = train_df['sentence'].apply(cal_text_len)
dev_df['text_len'] = dev_df['sentence'].apply(cal_text_len)


print(train_df['text_len'].value_counts())
print(dev_df['text_len'].value_counts())

print('-------------------')

label_list = config.label_list
print(label_list)
print('-------------------')

def make_label(label):
    return label_list.index(label)

train_df['num_label'] = train_df['label'].apply(make_label)
dev_df['num_label'] = dev_df['label'].apply(make_label)

train_df[['text', 'sentence', 'label', 'num_label']].to_csv(config.base_dir + 'train.csv', encoding='utf-8')
dev_df[['text', 'sentence', 'label', 'num_label']].to_csv(config.base_dir + 'dev.csv', encoding='utf-8')

print(train_df['num_label'])
print('-------------------')
print(train_df['num_label'].unique())
print('-------------------')
print(train_df['num_label'].value_counts())