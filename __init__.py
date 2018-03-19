# -*- coding: UTF-8 -*-
import pandas as pd # 数据分析
import numpy as np # 科学计算
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
if __name__=='__main__':
    data_train = pd.read_csv("train.csv") # 读入数据，将表格csv数据都城dataframe格式

    # 查看 data_train的信息
    #data_train.info()

    """
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890   # 891个实体，0 到 890
    Data columns (total 12 columns):    # 数据共有12列
    PassengerId    891 non-null int64   # 乘客id
    Survived       891 non-null int64   # 逃生
    Pclass         891 non-null int64   # 乘客等级
    Name           891 non-null object  # 乘客姓名
    Sex            891 non-null object  # 性别
    Age            714 non-null float64 # 年龄，只有714个乘客有年龄信息
    SibSp          891 non-null int64   # 堂兄弟/妹个数
    Parch          891 non-null int64   # 父母与小孩个数
    Ticket         891 non-null object  # 船票信息
    Fare           891 non-null float64 # 票价
    Cabin          204 non-null object  # 客舱，只有 204 个乘客有信息
    Embarked       889 non-null object  # 登船港口
    dtypes: float64(2), int64(5), object(5) 
    memory usage: 83.6+ KB
    """


    # fig = plt.figure()
    # fig.set(alpha = 0.2) # 设定图表颜色alpha参数
    # data_train.Pclass.value_counts().plot(kind='bar')  # 柱状图
    # plt.title('获救情况（1为获救）')    # 标题
    # plt.ylabel('人数')
    # plt.show()

    # X = data_train.Pclass.value_counts()
    # s = Series()
    # xk = X.keys()
    # xk = list(xk)
    # xk.sort()
    # fig= plt.figure()
    # plt.bar(xk,X[xk])
    # for ix in xk:
    #     plt.text(ix + 0.1, X[ix] + 0.05, '%d' % X[ix], ha='center', va='bottom')
    # plt.xlabel('等级')
    # plt.ylabel('人数')
    # plt.title('等级-人数图')
    # plt.show()

    """
    fig = plt.figure()
    fig.set(alpha = 0.2) # 设定图表颜色alpha参数
    
    plt.subplot2grid((2,3),(0,0))    # 在一张大图中分裂几个小图，下面的柱状图处于（0,0）位置
    data_train.Survived.value_counts().plot(kind='bar')  # 柱状图
    plt.title('获救情况（1为获救）')    # 标题
    plt.ylabel('人数')
    
    plt.subplot2grid((2,3),(0,1))
    data_train.Pclass.value_counts().plot(kind="bar")
    plt.title("人数")
    plt.ylabel("乘客等级分布")
    
    plt.subplot2grid((2,3),(0,2))
    plt.scatter(data_train.Survived,data_train.Age)
    plt.ylabel("年龄")
    plt.grid(b=True,which='major',axis='y')
    plt.title('按年龄看获救分布（1为获救）')
    
    plt.subplot2grid((2,3),(1,0), colspan=2)
    data_train.Age[data_train.Pclass == 1].plot(kind='kde')
    data_train.Age[data_train.Pclass == 2].plot(kind='kde')
    data_train.Age[data_train.Pclass == 3].plot(kind='kde')
    plt.xlabel(u"年龄")# plots an axis lable
    plt.ylabel(u"密度")
    plt.title(u"各等级的乘客年龄分布")
    plt.legend((u'头等舱', u'2等舱',u'3等舱'),loc='best') # sets our legend for our graph.
    
    plt.subplot2grid((2,3),(1,2))
    data_train.Embarked.value_counts().plot(kind='bar')
    plt.title(u"各登船口岸上船人数")
    plt.ylabel(u"人数")
    plt.show()
    """


    # 乘客等级的获救情况
    """
    Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
    Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
    df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
    df.plot(kind='bar', stacked=True)
    plt.title(u"各乘客等级的获救情况")
    plt.xlabel(u"乘客等级")
    plt.ylabel(u"人数")
    plt.show()
    """

    # 数据预处理
    # 冗余信息删除
    # 缺失值处理
    # 非数值型数据转化成数据值
    # 异常值处理




    from sklearn.ensemble import RandomForestRegressor
    df = data_train

    # 将缺失值很多的Cabin分成两种，一种是有值的，一种是无值的
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"

    # 使用随机森林将缺失的年龄属性补全
    age_df = df[['Age','Fare','Parch','SibSp','Pclass']] # 将这些属性取出来

    # 将乘客年龄分成两部分，已知和未知

    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # y 就是目标年龄
    y = known_age[:,0] # 取出第 0 行，也就是年龄
    X = known_age[:,1:] # X就是特征属性值，就是其他的属性值，比如Fare,Parch...

    rfr = RandomForestRegressor(random_state=0,n_estimators=2000,n_jobs= -1)
    rfr.fit(X,y) # 得到模型

    # 用模型得到结果
    predictedAges = rfr.predict(unknown_age[:,1::])

    # 定位到 df.Age.isnull() 位置，将预测到的值，赋值到原来的位置
    df.loc[(df.Age.isnull()),'Age'] = predictedAges

    # 将离散的无等级关系的数据做one-hot变换，将其前缀设为Cabin
    dummies_Cabin = pd.get_dummies(df['Cabin'],prefix='Cabin')
    dummies_Embarked = pd.get_dummies(df['Embarked'],prefix='Embarked')
    dummies_Sex = pd.get_dummies(df['Sex'],prefix='Sex')
    dummies_Pclass = pd.get_dummies(df['Pclass'],prefix='Pclass')

    # axis = 1 表示列
    df = pd.concat([df,dummies_Cabin,dummies_Embarked,dummies_Sex,dummies_Pclass],axis=1)
    # inplace 表示就地转换
    df.drop(['Pclass','Name','Sex','Ticket','Cabin','Embarked'],axis = 1,inplace=True)


    # 将Age和Fare归一化
    # 用 sklearn的 preprocessing模块做一个scaling
    import sklearn.preprocessing as preprocessing
    scaler = preprocessing.StandardScaler()

    # print(scaler)
    # scaler: StandardScaler(copy=True, with_mean=True, with_std=True)

    # 因为scaler.fit(arg)必须接收二维数据
    age_scale_param = scaler.fit(df['Age'].values.reshape(-1,1))
    dftmp = scaler.fit_transform(df['Age'].values.reshape(-1,1),age_scale_param)
    df['Age_scaled'] = dftmp[:,0]
    fare_scale_param = scaler.fit(df['Fare'].values.reshape(-1,1))
    dftmp = scaler.fit_transform(df['Fare'].values.reshape(-1,1),fare_scale_param)
    df['Fare_scaled'] = dftmp[:,0]

    # df.filter 将数据过滤，只取出想要的属性值
    train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    train_np = train_df.as_matrix()

    # y 也就是Survival结果
    y = train_np[:,0]

    # X即特征属性值
    X = train_np[:,1:]

    from sklearn import  linear_model

    # C ： 逆正则化强度，必须是一个整数，向支持向量机中，更小的值有更强的正则效果？
    # penalty ： 逻辑回归默认带了正则化项，有L1和L2两种正则化，默认是L2正则化
    #            其中当选择L1正则化时，只能选择liblinear
    #            当选择L2正则化时，可以选择使用牛顿法等方法
    # tol ： 停止的误差
    clf = linear_model.LogisticRegression(C=1.0,penalty='l1',tol=1e-6)

    from sklearn import cross_validation

    # 做简单的交叉验证

    # 将数据分成 5 份
    # print(cross_validation.cross_val_score(clf,X,y,cv = 5))


    from sklearn.ensemble import BaggingRegressor
    # 使用sklearn中的bagging进行模型融合
    # gg：用了多个 逻辑回归模型融合，提交的准确率反倒变低
    clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)

    # 训练模型
    clf.fit(X,y)

    # 查看各特征属性对模型的影响
    # print(pd.DataFrame({"columns":list(train_df.columns)[1:], "coef":list(clf.coef_.T)}))

    # 得到训练模型
    # print(clf)

    #>>>>>>>>>>>>>>>>>>>>>>>>>>> 读取测试数据，对测试数据做和训练数据同样的处理<<<<<<<<<<<<<<<<<<<<<<<

    data_test = pd.read_csv('test.csv')
    data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0

    # 用同样的 RandomForestRegressor 模型填上丢失的年龄
    tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
    null_age = tmp_df[data_test.Age.isnull()].as_matrix()

    # 根据特征属性X预测年龄并补上
    X = null_age[:, 1:]
    predictedAges = rfr.predict(X)
    data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges

    data_test.loc[ (data_test.Cabin.notnull()), 'Cabin' ] = "Yes"
    data_test.loc[ (data_test.Cabin.isnull()), 'Cabin' ] = "No"

    dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')
    dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
    dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
    dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')

    df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

    df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'].values.reshape(-1,1), age_scale_param)
    df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'].values.reshape(-1,1), fare_scale_param)

    test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

    # 使用模型预测，并保存结果
    predictions = clf.predict(test)
    result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
    result.to_csv("logistic_regression_predictions.csv", index = False)

