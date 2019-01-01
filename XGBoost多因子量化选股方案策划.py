import pandas as pd
from sklearn import preprocessing
import numpy as np
import pandas as pd
import numpy as np
#import xgboost as xgb
#from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation,metrics #Additional scklearn functions
from sklearn.grid_search import GridSearchCV #Perforing grid search
import matplotlib.pylab as plt
#%matplotlib inline
from matplotlib.pylab import rcParams
import xlrd,xlsxwriter
from pandas import DataFrame





# 其中每个文件中有多个sheet，需要将其全部合并
 
#设置要合并的所有文件
allxls =['/Users/jing/Downloads/数据/股票指标1.xlsx','/Users/jing/Downloads/数据/股票指标2.xlsx','/Users/jing/Downloads/数据/股票指标3.xlsx']
#设置合并到的文件
endxls ="/Users/jing/Downloads/数据/股票指标/endxls.xlsx"
 
#打开表格
def open_xls(file):
    try:
        fh=xlrd.open_workbook(file)
        return fh
    except Exception as e:
        print(str("打开出错，错误为："+e))
 
#获取所有sheet
def getsheet(fh):
    return fh.sheets()
 
#读取某个sheet的行数
def getnrows(fh,sheet):
    table=fh.sheets()[sheet]
    content=table.nrows
    return content
 
#读取某个文件的内容并返回所有行的值
def getfilect(fh,fl,shnum):
    fh=open_xls(fl)
    df = pd.read_excel(fl,sheet_by_name_name(shname[shnum])
    table=fh.sheet_by_name(shname[shnum])
    print (table.row_values(0))
    print (df.columns)
    #num=getnrows(fh,shnum)
    #lenrvalue=len(rvalue)
    #for row in range(0,num):
        #rdata=table.row_values(row)
        #rvalue.append(rdata)
    #print(rvalue[lenrvalue:])
    #filevalue.append(rvalue[lenrvalue:])
    #return filevalue
 
#存储所有读取的结果
filevalue=[]
#存储一个标签的结果
svalue=[]
#存储一行结果
rvalue=[]
#存储各sheet名
shname=[]
 
#读取第一个待读文件，获得sheet数
fh=open_xls(allxls[0])
sh=getsheet(fh)
x=0
for sheet in sh:
    shname.append(sheet.name)
    svalue.append([])
    x+=1
#依次读取各sheet的内容
#依次读取各文件当前sheet的内容
for shnum in range(0,x):
    for fl in allxls:
        print("正在读取文件："+str(fl)+"的第"+str(shnum)+"个标签的…")
        filevalue=getfilect(fh,fl,shnum)
    #svalue[shnum].append(filevalue)
    #print(svalue[0])
    #print(svalue[1])
#由于apped具有叠加关系，分析可得所有信息均在svalue[0][0]中存储
#svalue[0][0]元素数量为sheet标签数(sn)*文件数(fn)
#sn=x
#fn=len(allxls)
#endvalue=[]
 
#设置一个函数专门获取svalue里面的数据，即获取各项标签的数据
#def getsvalue(k):
    #for z in range(k,k+fn):
        #endvalue.append(svalue[0][0][z])
    #return endvalue
 
#打开最终写入的文件
#wb1=xlsxwriter.Workbook(endxls)
#创建一个sheet工作对象
#ws=wb1.add_worksheet()
#polit=0
#linenum=0
#依次遍历每个sheet中的数据
#for s in range(0,sn*fn,fn):
    #thisvalue=getsvalue(s)
    #tvalue=thisvalue[polit:]
    #将一个标签的内容写入新文件中
    #for a in range(0,len(tvalue)):
        #for b in range(0,len(tvalue[a])):
            #for c in range(0,len(tvalue[a][b])):
                #print(linenum)
                #print(c)
                #data=tvalue[a][b][c]
                #ws.write(linenum,c,data)
            #linenum+=1
    #叠加关系，需要设置分割点
    #polit=len(thisvalue)
    

#finacial_data = DataFrame()
#l = []
#h = []
#for i in range(len(sheets)):
    #df= pd.read_excel(excel_name,sheet_name = i)
    #print (df.columns)
    ##for r in list(df.columns):
        ##if r not in ('证券代码','证券简称'):
            #h.append(r)
    #df_T = df.T
    #df_T.index = h
    #df_T1 = df_T['date_q'].to_period('Q')
    #df_T1['date_q'] = list(df_T1.index)
    #print (df_T.index)
#for c in list(df.证券代码):
    #c1= c + sheets[i]
    #l.append(c1)
    #df_t = df.T
    #df_t.columns = l


    
    #print (df_t.columns)
    #finacial_data = pd.merge(finacial_data,df,on='date',how='inner')
    #finacial_data = finacial_data.dropna()
    #print(len(finacial_data))

#gz12 = pd.merge(finacial_data01,finacial_data02,on=['证券代码','date'),how='inner')
#gz123 = pd.merge(gz12,finacial_data03,on=['证券代码','date'),how='inner')
#gz_hg = pd.merge(gz123,hg,on=['证券代码','date'),how='inner')
#gz_hg.to_csv(r'/Users/jing/Downloads/数据/MODEL/xgboost_data.csv')



#finacial_data = pd.read_excel(r'/Users/jing/Downloads/数据/股票指标3.xlsx')

#finacial_data.index.name = 'date'
#finacial_data.columns.name = '证券代码'
#print (col_list)
#finacial_data.index = list(finacial_data.datetime.datetime(2017, 12, 31, 0, 0))
#finacial_data1 = finacial_data.to_period('Q')
#finacial_data1['date_q'] = list(finacial_data1.index)

#finacial_data2 = finacial_data.dropna(axis=1,how='all').T
#print (finacial_data2)
#c_l1 = {b:b.split('/')[0]+'_'+b.split('/')[1] for b in list(finacial_data2.columns)\ if len(b.split('/'))>=2}
#print (c_l1)
#c_l2 = {b:b.split('/')[0]+'_'+b.split('/')[1] for b in list(finacial_data2.columns)\ if len(b.split('/'))>=2}
#finacial_data3 = finacial_data2.rename(columns=c_l1)
#print (finacial_data3)
#finacial_data3 = finacial_data3.rename(columns=c_l2)

#col_list = []
#col_list.extend(list(finacial_data2.columns))
#print (col_list)
#finacial_data2.to_csv(r'/Users/jing/Downloads/finacial_data3.csv')

#fin_d1 = pd.read_excel(r'/Users/jing/Downloads/数据/股票指标2.xlsx')
#fin_d2 = pd.read_excel(r'/Users/jing/Downloads/数据/股票指标1.xlsx')
#fin_d = pd.concat([fin_d1,fin_d2],ignore_index=True)
#col_list.extend(list)
#hg = pd.read_excel(r'/Users/jing/Downloads/数据/宏观数据.xlsx')
#hg.columns
#col_list 


def modelfit(alg,dtrain,predictors,useTrainCV=True,cv_folds=5,early_stopping_rounds=50):
    target = 'label'
    if useTrainCV:
          xgb_param = alg.get_xgb_params()
          xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
          cvresult = xgb.cv(xgb_param,xgtrain,num_boost_round=alg.get_params()['n_estimators'],nfold=cv_folds,metrics='auc',early_stopping_rounds=early_stopping_rounds,verbose_eval=True)
          alg.set_params(n_estimators=cvresult.shape[0])

    #Fit the algorithm on the data
    alg.fit(dtrain[predictors],dtrain['label'],eval_metric='auc')

    #Predict training set:
    dtrain_predictions = alg.predic(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]

    #print model report:
    print("\nModel Report")
    print("Accuracy:%.4g"% metrics.accuracy_score(dtrain['label'].values,dtrain_predictions))
    print("AUC Score(Train):%f"% metrics.roc_auc_score(dtrain['label'],dtrain_predprob))

    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)[:200]
    import_f = feat_imp.index
    #print(feat_imp)
    feat_imp.plot(kind='bar',title='Feature Importances')
    #plt.ylabel('Feature Importance Score')
    import matplotlib
    zhfont1 = matplotlib.font_manager.FontProperties(fname='/Users/jing/Downloads/simhei.ttf')
    plt.ylabel('特征重要性分数',fontproperties=zhfont1)
    plt.title('特征重要性',fontproperties=zhfont1)
