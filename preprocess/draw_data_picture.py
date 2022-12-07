import pandas as pd
import matplotlib.pyplot as plt
from config import Config
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签


def cal_text_len(row):
    row_len = len(row)
    if row_len < 256:
        return '0-256'
    elif row_len < 384:
        return '256-384'
    elif row_len < 512:
        return '384-512'
    elif row_len < 1024:
        return '512-1024'
    else:
        return '1024++'


config = Config()

train_df = pd.read_csv(config.base_dir + 'train.csv', encoding='utf8')

train_df['text_len'] = train_df['text'].apply(cal_text_len)

x_y_list = dict(train_df['text_len'].value_counts())

x_name = ['0-256', '256-384', '384-512', '512-1024','1024++', ]
y = [x_y_list[i] for i in x_name]

print(sum(y)) # 数据数量
# 画出条形图
#  color=['b','r','g','y','c','m','y','k','c','g','g']
plt.bar(x_name, y, color=['b', 'r', 'g', 'y', 'c', 'm','k',])
plt.xlabel('长度')
plt.ylabel('数量/条')
plt.title('原始数据文本长度数量分布图')
plt.savefig(config.base_dir + "cls_source.jpeg",  dpi=600)
plt.show()

plt.pie(x=y,  # 绘图数据
        labels=x_name,  # 添加教育水平标签
        autopct='%.1f%%'  # 设置百分比的格式，这里保留一位小数
        )
plt.title('原始数据文本长度分布占比图')
plt.savefig(config.base_dir + "cls_source_pie.jpeg", dpi=600)
plt.show()
