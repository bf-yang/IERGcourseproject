import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import Counter

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


#索引1为activity_id
#索引[4:16)/[21,33)/[38,50)分别为3个IMU的3D-acc1,3D-acc2,3D-gyro,3D-magn(共36种特征)
loc = [1] + [*range(4,16)] + [*range(21,33)] + [*range(38,50)]

def window(data, label, size, stride):
    '''将数组data和label按照滑窗尺寸size和stride进行切割'''
    x, y = [], []
    for i in range(0, len(label), stride):
        if i+size < len(label): #不足一个滑窗大小的数据丢弃

            l = set(label[i:i+size])
            if len(l) > 1 or label[i] == 0: #当一个滑窗中含有包含多种activity或者activity_id为0（即属于其他动作），丢弃
                continue
            elif len(l) == 1:
                x.append(data[i: i + size, :])
                y.append(label[i])

    return x, y

def generate(window_size, step, subject_list):
    '''生成训练样本X和对应标签Y'''
    X, Y = [], []
    # 遍历9个subject文件
    for i in subject_list:
    # for i in range(1, 10):
        total = pd.read_csv('./PAMAP2_Dataset/Protocol/subject10' + str(i) + '.dat', header=None, sep=' ', usecols=loc).values
        total = total[~np.isnan(total).any(axis=1), :]  #去除NaN
        data = total[:, 1:]
        label = total[:, 0].reshape(-1)

        # 调用window函数进行滑窗处理
        x, y = window(data, label, window_size, step)
        X += x
        Y += y

    # 将索引从0开始依次编号
    # {1,2,3,4,5,6,7,12,13,16,17,24} -> {0:11}
    cate_idx = list(Counter(Y).keys())
    cate_idx.sort()
    for i in range(len(Y)):
        Y[i] = cate_idx.index(Y[i])

    return X, Y

def category(X, Y):
    '''按照种类分类动作'''
    result = [[] for i in range(len(list(Counter(Y).keys())))]  #result对应的索引即标签
    for step, y in enumerate(Y):
        result[y].append(X[step])
    return result

def split(result, test_size):
    '''划分数据集
    test_size:测试集样本数量占比'''
    x_train, x_test, y_train, y_test = [], [], [], []
    for i, data in enumerate(result):
        label = [i for n in range(len(data))]
        x_train_, x_test_, y_train_, y_test_ = train_test_split(data, label, test_size=test_size, shuffle=True)
        x_train.extend(x_train_)
        y_train.extend(y_train_)
        x_test.extend(x_test_)
        y_test.extend(y_test_)
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    # # split datset into train + test 
    # subject_list_train = [1,2,3,4,5,6]
    # subject_list_test = [7,8,9]

    # x_train, y_train = generate(171, 85, subject_list_train)
    # x_test, y_test = generate(171, 85, subject_list_test)
    # print(len(y_train),len(y_test))
    
    # # np.save('./Datasets/x_train', x_train)
    # # np.save('./Datasets/x_test', x_test)
    # # np.save('./Datasets/y_train', y_train)
    # # np.save('./Datasets/y_test', y_test)
    
    
    # split datset into pre-train + fine-tune + test 
    subject_list_pretrain = [1,2,3,4]
    subject_list_finetune = [5,6,7]
    subject_list_test = [8,9]

    x_pretrain, y_pretrain = generate(171, 85, subject_list_pretrain)
    x_finetune, y_finetune = generate(171, 85, subject_list_finetune)
    x_test, y_test = generate(171, 85, subject_list_test)

    print(len(y_pretrain),len(y_finetune),len(y_test))
    
    np.save('./Datasets_ptft/x_pretrain', x_pretrain)
    np.save('./Datasets_ptft/x_finetune', x_finetune)
    np.save('./Datasets_ptft/x_test', x_test)
    np.save('./Datasets_ptft/y_pretrain', y_pretrain)
    np.save('./Datasets_ptft/y_finetune', y_finetune)
    np.save('./Datasets_ptft/y_test', y_test)
