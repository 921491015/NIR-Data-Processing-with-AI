import numpy as np
from sklearn import preprocessing
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, recall_score, classification_report, accuracy_score, make_scorer
import pandas as pd
import matplotlib.pyplot as plt
from pretty_confusion_matrix import pp_matrix
import joblib
import matplotlib
matplotlib.use('TkAgg')  # 绘图必须添加这个，否则出对象错误
from sklearn.model_selection import train_test_split
import time

def PLS_DA(array,pc):
    grape_data = pd.read_csv(
        r'C:\Users\92149\Desktop\varietal_classification_raw_spectra_PONE-D-15-379\1_vergalijo_20_varieties.csv')
    y = grape_data.iloc[:, 100:101].values  # 标签
    y = np.array(y)
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    x = array
    # 然后对y进行转换
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    y_train = pd.get_dummies(y_train)  # 二进制编码，否则不能进行PLS回归
    plsmodel = PLSRegression(n_components=pc)
    plsmodel.fit(x_train, y_train)
    y_pred = plsmodel.predict(x_test)
    y_pred1 = np.array([np.argmax(i) for i in y_pred])
    score = accuracy_score(y_test, y_pred1)
    print(str(score))

    # 寻找最佳主城分数
    best_parameters = {}
    best_score = 0
    for pc1 in range(1, 100):
        plsmodel = PLSRegression(n_components=pc1)
        plsmodel.fit(x_train, y_train)
        y_pred = plsmodel.predict(x_test)
        y_pred2 = np.array([np.argmax(i) for i in y_pred])
        scorefor = accuracy_score(y_test, y_pred2)
        if scorefor > best_score:
            best_score = scorefor
            best_parameters = {'pc param': pc1}
    print('最佳测试分数:', best_score)
    print('最佳测试下参数(主成分数):', best_parameters)

# 脚本可独自运行，也可以提供外部函数
if __name__ == '__main__':
    '''
    def logscore(y_test1,pred):#用来规定自己的评价函数
        pred1=pred
        test1=y_test1
        pred1 = np.array([np.argmax(i) for i in pred1])
        score=accuracy_score(test1,pred1)
    
    loss  = make_scorer(logscore(), greater_is_better=False)
    score = make_scorer(logscore(), greater_is_better=True)
    '''
    #读取特征矩阵
    grape_data = pd.read_csv(
        r'C:\Users\92149\Desktop\varietal_classification_raw_spectra_PONE-D-15-379\1_vergalijo_20_varieties.csv')
    y = grape_data.iloc[:, 100:101].values  # 标签
    y = np.array(y)
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    test_y=y
    grape_data = grape_data.iloc[:, :100]
    grape_array = np.array(grape_data)
    x = grape_array

    #然后对y进行转换
    y = pd.get_dummies(y)#二进制编码，否则不能进行PLS回归

    #使用循环迭代最佳主成分数，用来找到最佳分类
    '''
    best_cm=[]
    best_parameters = {}
    best_score = 0
    for pc in range(1,100):
        plsmodel = PLSRegression(n_components=pc)
        plsmodel.fit(x,y)
        x_test = x
        y_pred = plsmodel.predict(x_test)
        y_pred = np.array([np.argmax(i) for i in y_pred])
        score = accuracy_score(test_y,y_pred)
        if score > best_score:
            best_score = score
            best_parameters = {'pc param':pc}
            best_cm=confusion_matrix(test_y,y_pred)
            best_pred=y_pred
            best_model=plsmodel
            bestpc=pc
    print('最佳测试分数:',best_score)
    print('最佳测试下参数(主成分数):',best_parameters)
    joblib.dump(best_model, 'PLSDAwithACC={}PARAM={}.pkl'.format(best_score,bestpc))
    '''
    start=time.time()
    plsmodel = PLSRegression(n_components=83)
    plsmodel.fit(x, y)
    x_test = x
    y_pred = plsmodel.predict(x_test)
    y_pred1 = np.array([np.argmax(i) for i in y_pred])
    y_pred2=plsmodel.predict(x_test).flatten()
    score = accuracy_score(test_y, y_pred1)
    print(str(score))
    print(plsmodel.x_scores_)
    end=time.time()
    print('running time:'+str(end-start))

    '''
    labels = le.inverse_transform(test_y)
    predicted = le.inverse_transform(best_pred)
    y1 = le.inverse_transform(test_y)
    cm = confusion_matrix(labels, predicted)
    cm = pd.DataFrame(cm, index=np.unique(y1), columns=np.unique(y1))
    cm.index.name = '实际种类'
    cm.columns.name = '预测种类'
    # fig, ax = plt.subplots(figsize=(15, 10))
    plt.rcParams.update({'font.size': 7})
    cmap = "Pastel1"
    pp_matrix(cm, cmap=cmap, figsize=[15, 9])
    '''