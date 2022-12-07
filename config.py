# -*- coding: utf-8 -*-
# @Time    : 2022/11/7 14:15
# @Author  : zhangwei
# @Email   : 751148173@qq.com
# @File    : config.py
# @Software: PyCharm


class Config(object):
    def __init__(self):
        self.base_dir = 'C:\\Users\\zhangwei\\Desktop\\文本分类\\data\\'  # 数据路径
        self.save_model = 'C:\\Users\\zhangwei\\Desktop\\文本分类\\'  # 模型路径
        self.result_file = 'C:\\Users\\zhangwei\\Desktop\\文本分类\\'
        self.label_list = ['体育', '娱乐', '家居', '房产', '教育', '时尚', '时政', '游戏', '科技', '财经']

        self.warmup_proportion = 0.05 # 慢热学习比例，用来保证前面0.05训练步伐的学习率较低，避免模型在训练的开始由于其随机初始化的权重导致的训练震荡，而后学习率再缓慢趋向于之前设置的学习率。
        self.use_bert = True
        self.pretrainning_model = 'roberta'
        self.embed_dense = 512

        self.decay_rate = 0.5  # 下游网络结构的学习率衰减参数，因为网络结构的学习率随着训练进程的衰减有助于模型更好地拟合数据。

        self.train_epoch = 4  # 训练迭代次数

        self.learning_rate = 1e-4  # 下接结构学习率
        self.embed_learning_rate = 5e-5  # 预训练模型学习率 2e-5 3e-5

        if self.pretrainning_model == 'roberta':
            model = 'C:\\Users\\zhangwei\\Desktop\\文本分类\\model\\roberta\\'  # 中文roberta-base
        elif self.pretrainning_model == 'bert':
            model = 'C:\\Users\\zhangwei\\Desktop\\文本分类\\model\\bert\\'  # 中文nezha-base
        elif self.pretrainning_model == 'albert':
            model = 'C:\\Users\\zhangwei\\Desktop\\文本分类\\model\\albert\\'  # 中文nezha-base
        elif self.pretrainning_model == 'nezha':
            model = 'C:\\Users\\zhangwei\\Desktop\\文本分类\\model\\nezha-cn-base\\'  # 中文nezha-base
        else:
            raise KeyError('albert nezha roberta bert bert_wwm is need')
        self.cls_num = 10
        self.sequence_length = 512
        self.batch_size = 1

        self.model_path = model

        self.bert_file = model + 'pytorch_model.bin'
        self.bert_config_file = model + 'config.json'
        self.vocab_file = model + 'vocab.txt'

        self.use_origin_bert = 'weight'  # 'ori':使用原生bert, 'dym':使用动态融合bert,'weight':初始化12*1向量
        self.is_avg_pool = 'mean'  # dym, max, mean, cls
        self.model_type = 'bilstm'  # bilstm; bigru

        self.rnn_num = 2
        self.flooding = 0
        self.embed_name = 'bert.embeddings.word_embeddings.weight'  # 词
        self.restore_file = None
        self.gradient_accumulation_steps = 1
        # 模型预测路径
        self.checkpoint_path = "C:\\Users\\zhangwei\\Desktop\\文本分类\\runs_1\\1670296898\\model_0.9726_0.9726_0.9726_2336.bin"
        # self.checkpoint_path = "/home/wangzhili/chile/tsing_hua_code/data/Savemodel/runs_0/1624199990/model_dist.bin"

        """
        实验记录
        """
