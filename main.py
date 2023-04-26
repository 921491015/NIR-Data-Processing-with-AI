import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')  # 绘图必须添加这个，否则出对象错误
import matplotlib.pyplot as plt
import paint_raw_data
import preprocessing_data
import SVM_processing
import PLS_DA_processing
import ANN_processing
import os

def print_hi(name):
    print(f'Hi, {name}')

if __name__ == '__main__':
    print_hi('PyCharm')
    '''数据准备-预处理'''
    raw_array = paint_raw_data.raw_data()
    snv_array = preprocessing_data.preprocess_SNV()  # 数据预处理SNV
    detrend_array = preprocessing_data.detrending()  # 数据预处理去趋势
    S_plus_D_array = preprocessing_data.SVN_plus_D(detrend_array)  # SVN+D预处理
    SD_plus_SGD2W5_array = preprocessing_data.preprocess_SG_D2W5(S_plus_D_array)  # SVN+D+SG预处理
    SD_plus_SGD2W11_array = preprocessing_data.preprocess_SG_D2W11(S_plus_D_array)  # SVN+D+SG预处理
    SD_plus_SGD1W5_array = preprocessing_data.preprocess_SG_D1W5(S_plus_D_array)  # SVN+D+SG预处理
    SD_plus_SGD1W11_array = preprocessing_data.preprocess_SG_D1W11(S_plus_D_array)  # SVN+D+SG预处理
    RAW_plus_SGD2W5_array = preprocessing_data.preprocess_SG_D2W5(raw_array)  # SG预处理
    RAW_plus_SGD2W11_array = preprocessing_data.preprocess_SG_D2W11(raw_array)  # SG预处理
    RAW_plus_SGD1W5_array = preprocessing_data.preprocess_SG_D1W5(raw_array)  # SG预处理
    RAW_plus_SGD1W11_array = preprocessing_data.preprocess_SG_D1W11(raw_array)  # SG预处理
    array_spt = []
    array_spt.append(raw_array)
    array_spt.append(detrend_array)
    array_spt.append(snv_array)
    array_spt.append(S_plus_D_array)
    array_spt.append(SD_plus_SGD1W11_array)
    array_spt.append(SD_plus_SGD1W5_array)
    array_spt.append(SD_plus_SGD2W5_array)
    array_spt.append(SD_plus_SGD2W11_array)
    array_spt.append(RAW_plus_SGD1W11_array)
    array_spt.append(RAW_plus_SGD1W5_array)
    array_spt.append(RAW_plus_SGD2W11_array)
    array_spt.append(RAW_plus_SGD2W5_array)
    '''算法分类'''
    # SVM_processing.SVM_data(10,2,SD_plus_SGD2W5_array)#打印某参数SVM下的分类结果准确率及绘制混淆矩阵
    # ANN_processing.ANN_data(SD_plus_SGD2W11_array, (100), 0.0005)
    # PLS_DA_processing.PLS_DA(SD_plus_SGD2W11_array,85)
    # os.system("python CNN_processing.py")
    '''绘制'''
    # paint_raw_data.paint_raw_data()#绘制原始数据图像
    # preprocessing_data.paint_preprocess_data(snv_array,detrend_array,S_plus_D_array,SD_plus_SGD2W5_array)#绘制预处理过后的数据图像
    # preprocessing_data.paint_preprocess_data_s(SD_plus_SGD2W5_array)  # 绘制SG预处理过后的数据图像

    '''循环测试'''
    '''
    wjq = 0
    for ceshi in array_spt:
        wjq = wjq + 1
        print(wjq)
        PLS_DA_processing.PLS_DA(ceshi, 85)
    '''