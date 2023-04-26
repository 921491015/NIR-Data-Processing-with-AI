import pandas as pd
import numpy as np
import matplotlib
from matplotlib import ticker

matplotlib.use('TkAgg')  # 绘图必须添加这个，否则出对象错误
import matplotlib.pyplot as plt
from PIL import Image

def paint_raw_data():
    #读取数据
    grape_data = pd.read_csv(
        r'C:\Users\92149\Desktop\varietal_classification_raw_spectra_PONE-D-15-379\1_vergalijo_20_varieties.csv')
    #切分数据，以第一行数据为例准备绘图
    paintgrape_data = grape_data.iloc[:, :]
    paintgrape_array = np.array(paintgrape_data)  # 转换DataFrame为array数组
    paintgrape_array1 = np.delete(paintgrape_array, 100, 1) #去除品种信息
    paintgrape_array2 = np.transpose(paintgrape_array1)#经测试，需要转置维度
    #提取列标签绘图
    axislist = grape_data.columns.tolist()
    axislist.pop(100)
    axisarray = np.array(axislist)
    #开始绘图
    plt.style.use("ggplot")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False#解决中文乱码
    fig_raw,ax_raw=plt.subplots(figsize=(15,7))#figsize参数控制图形的大小
    ax_raw.plot(axisarray,paintgrape_array2)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(7))#改善横坐标个数，看清楚
    #ax_raw.set_xticks([1500,1600,1700,1800,1900,2000])
    ax_raw.set_title("原始数据曲线")  # 添加标题
    ax_raw.set_xlabel("波长")  # 添加x轴标签
    ax_raw.set_ylabel(r"Log(1/R)")  # 添加y轴标签
    ax_raw.legend()  # 显示图例
    #plt.show()
    plt.savefig(r"C:\Users\92149\Desktop\varietal_classification_raw_spectra_PONE-D-15-379\picture\line of raw data.png")
    img = Image.open(r'C:\Users\92149\Desktop\varietal_classification_raw_spectra_PONE-D-15-379\picture\line of raw data.png')
    img.show()

def raw_data():
    grape_data = pd.read_csv(
        r'C:\Users\92149\Desktop\varietal_classification_raw_spectra_PONE-D-15-379\1_vergalijo_20_varieties.csv')

    grape_data = grape_data.iloc[:, :100]
    grape_array = np.array(grape_data)
    return grape_array

