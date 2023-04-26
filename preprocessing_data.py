import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')  # 绘图必须添加这个，否则出对象错误
from matplotlib import ticker
import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal

#返回SNV处理后的矩阵
def preprocess_SNV():
    grape_data = pd.read_csv(
        r'C:\Users\92149\Desktop\varietal_classification_raw_spectra_PONE-D-15-379\1_vergalijo_20_varieties.csv')

    grape_data = grape_data.iloc[:, :100]
    grape_array=np.array(grape_data)
    temp1=grape_array.mean(axis=1)#算出400行每一行的均值
    temp2=np.tile(temp1, (grape_array.shape[1],1))#塑型，复制
    temp3=np.transpose(temp2)
    temp4=np.std(grape_array,axis=1)#计算出400行每一行的标准差
    temp5=np.tile(temp4,(grape_array.shape[1],1))#塑型，复制
    temp6 = np.transpose(temp5)
    grape_snv_array=(grape_array-temp3)/temp6
    return grape_snv_array

#返回去趋势的矩阵
def detrending():
    grape_data = pd.read_csv(
        r'C:\Users\92149\Desktop\varietal_classification_raw_spectra_PONE-D-15-379\1_vergalijo_20_varieties.csv')

    grape_data = grape_data.iloc[:, :100]
    grape_array = np.array(grape_data)
    detrend_grape_array=signal.detrend(grape_array,axis=1,type='linear',bp=0,overwrite_data=False)#去趋势，线性趋势
    return detrend_grape_array

#返回去趋势和SNV处理后的矩阵，先进行去趋势，后进行SNV处理
def SVN_plus_D(detrending_array):
    Detrend_array=detrending_array
    temp1 = Detrend_array.mean(axis=1)  # 算出400行每一行的均值
    temp2 = np.tile(temp1, (Detrend_array.shape[1], 1))  # 塑性，复制
    temp3 = np.transpose(temp2)
    temp4 = np.std(Detrend_array, axis=1)  # 计算出400行每一行的标准差
    temp5 = np.tile(temp4, (Detrend_array.shape[1], 1))  # 塑性，复制
    temp6 = np.transpose(temp5)
    grape_sd_array = (Detrend_array - temp3) / temp6
    return grape_sd_array

#返回经过SG滤波处理后的数据：D2W5
def preprocess_SG_D2W5(grape_sd_array):

    S_D_array=grape_sd_array
    raw_SG_array=signal.savgol_filter(S_D_array, 5, 2,axis=1,deriv=2,mode='nearest')
    return raw_SG_array

#返回经过SG滤波处理后的数据：D2W5
def preprocess_SG_D2W11(grape_sd_array):

    S_D_array=grape_sd_array
    raw_SG_array=signal.savgol_filter(S_D_array, 11, 2,axis=1,deriv=2,mode='nearest')
    return raw_SG_array

def preprocess_SG_D1W5(grape_sd_array):

    S_D_array=grape_sd_array
    raw_SG_array=signal.savgol_filter(S_D_array, 5, 2,axis=1,deriv=1,mode='nearest')
    return raw_SG_array

def preprocess_SG_D1W11(grape_sd_array):

    S_D_array=grape_sd_array
    raw_SG_array=signal.savgol_filter(S_D_array, 11, 2,axis=1,deriv=1,mode='nearest')
    return raw_SG_array

# 主要是绘图函数
def paint_preprocess_data(snv_array,detrend_array,s_plus_d_array,svn_sg_array):
    grape_snv_array=snv_array#传snv矩阵
    grape_D_array=detrend_array#传D处理矩阵
    grape_SD_array=s_plus_d_array#传SNV+D矩阵
    grape_SD_SG_array=svn_sg_array#传SNV+D+SG矩阵
    # 读取数据
    grape_data = pd.read_csv(
        r'C:\Users\92149\Desktop\varietal_classification_raw_spectra_PONE-D-15-379\1_vergalijo_20_varieties.csv')
    # 切分数据，以第一行数据为例准备绘图
    paintgrape_data = grape_data.iloc[:1, :]
    paintgrape_array = np.array(paintgrape_data)  # 转换DataFrame为array数组
    paintgrape_array1 = np.delete(paintgrape_array, 100, 1)  # 去除品种信息
    paintgrape_array2 = np.transpose(paintgrape_array1)  # 经测试，需要转置维度
    paintgrape_snv_array=grape_snv_array[:1,:]
    paintgrape_snv_array1=np.transpose(paintgrape_snv_array)
    paintgrape_D_array=grape_D_array[:1,:]
    paintgrape_D_array1=np.transpose(paintgrape_D_array)
    paintgrape_SD_array=grape_SD_array[:1,:]
    paintgrape_SD_array1=np.transpose(paintgrape_SD_array)
    paintgrape_SD_SG_array=grape_SD_SG_array[:1,:]
    paintgrape_SD_SG_array1=np.transpose(paintgrape_SD_SG_array)
    # 提取列标签绘图
    axislist = grape_data.columns.tolist()
    axislist.pop(100)
    axisarray = np.array(axislist)
    # 开始绘图
    plt.style.use("ggplot")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # 解决中文乱码
    fig_raw, ax_raw = plt.subplots(figsize=(15, 7))  # figsize参数控制图形的大小
    ax_raw.plot(axisarray, paintgrape_array2, color='blue', label='原始数据')
    ax_raw.plot(axisarray,paintgrape_snv_array1,color='red',label='正态变换处理后数据')
    ax_raw.plot(axisarray, paintgrape_D_array1, color='green', label='去趋势处理后数据')
    ax_raw.plot(axisarray, paintgrape_SD_array1, color='purple', label='去趋势和正态变换处理后数据')
    ax_raw.plot(axisarray, paintgrape_SD_SG_array1, color='black', label='去趋势、正态变换和SG滤波处理后数据')
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(7))  # 改善横坐标个数，看清楚
    ax_raw.set_title("预处理输曲线对比")  # 添加标题
    ax_raw.set_ylabel(r"Log(1/R)")  # 添加x轴标签
    ax_raw.set_xlabel("波长")  # 添加y轴标签
    ax_raw.legend()  # 显示图例
    # plt.show()
    plt.savefig(
        r"C:\Users\92149\Desktop\varietal_classification_raw_spectra_PONE-D-15-379\picture\line of processing data.png")
    img = Image.open(
        r'C:\Users\92149\Desktop\varietal_classification_raw_spectra_PONE-D-15-379\picture\line of processing data.png')
    img.show()

def paint_preprocess_data_s(svn_sg_array):
    grape_SD_SG_array=svn_sg_array#传SNV+D+SG矩阵
    # 读取数据
    grape_data = pd.read_csv(
        r'C:\Users\92149\Desktop\varietal_classification_raw_spectra_PONE-D-15-379\1_vergalijo_20_varieties.csv')
    # 切分数据，以第一行数据为例准备绘图
    paintgrape_SD_SG_array=grape_SD_SG_array[:1,:]
    paintgrape_SD_SG_array1=np.transpose(paintgrape_SD_SG_array)
    # 提取列标签绘图
    axislist = grape_data.columns.tolist()
    axislist.pop(100)
    axisarray = np.array(axislist)
    # 开始绘图
    plt.style.use("ggplot")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # 解决中文乱码
    fig_raw, ax_raw = plt.subplots(figsize=(15, 7))  # figsize参数控制图形的大小
    ax_raw.plot(axisarray, paintgrape_SD_SG_array1, color='black', label='去趋势、正态变换和SG滤波处理后数据')
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(7))  # 改善横坐标个数，看清楚
    ax_raw.set_title("经过SG+S+D处理曲线")  # 添加标题
    ax_raw.set_ylabel(r"Log(1/R)")  # 添加y轴标签
    ax_raw.set_xlabel("波长")  # 添加x轴标签
    ax_raw.legend()  # 显示图例
    # plt.show()
    plt.savefig(
        r"C:\Users\92149\Desktop\varietal_classification_raw_spectra_PONE-D-15-379\picture\line of SG+S+D data.png")
    img = Image.open(
        r'C:\Users\92149\Desktop\varietal_classification_raw_spectra_PONE-D-15-379\picture\line of SG+S+D data.png')
    img.show()