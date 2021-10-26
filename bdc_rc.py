import streamlit as st
import pickle
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import sklearn
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import random
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import sklearn.model_selection as model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler

#应用标题
st.title('A machine learning-based predictive model for predicting relapse in patients with bile duct cancer')

# conf
st.sidebar.markdown('## Variables')
HGB = st.sidebar.slider("HGB", 0, 200, value=100, step=1)
#Diameter_G = st.sidebar.selectbox('Diameter.G',('<5cm','5-10cm','>10cm'),index=0)
#T = st.sidebar.selectbox("T stage",('T1','T2','T3','TX'))
#M = st.sidebar.selectbox("M stage",('M0','M1'))
lymphonode = st.sidebar.selectbox("Lymphonode metastases",('No','Yes'),index=0)
cirrhosis = st.sidebar.selectbox("Cirrhosis",('No','Yes'),index=0)
vascularinvasion = st.sidebar.selectbox("Vascularinvasion",('No','Yes'),index=0)
steatosis = st.sidebar.selectbox("Steatosis",('No','Yes'),index=0)
#Bone_metastases = st.sidebar.selectbox("Bone metastases",('No','Yes'))
#Lung_metastases = st.sidebar.selectbox("Lung metastases",('No','Yes'))

# str_to_int

map = {'<5cm':0,'5-10cm':1,'>10cm':2,'No':0,'Yes':1,}
#map = {'White':0,'Black':1,'Other':2,'T1':0,'T2':1,'T3':2,'TX':3,'M0':0,'M1':1,'NX':2,'No':0,'Yes':1,}
#Age =map[Age]

lymphonode =map[lymphonode]
cirrhosis =map[cirrhosis]
vascularinvasion =map[vascularinvasion]
steatosis =map[steatosis]
#Radiation =map[Radiation]
#Chemotherapy =map[Chemotherapy]
#Bone_metastases =map[Bone_metastases]
#Lung_metastases =map[Lung_metastases]

# 数据读取，特征标注
thyroid_train = pd.read_csv('train.csv', low_memory=False)
thyroid_train['recur'] = thyroid_train['recur'].apply(lambda x : +1 if x==1 else 0)
#thyroid_test = pd.read_csv('test1.csv', low_memory=False)
#thyroid_test['infection'] = thyroid_test['infection'].apply(lambda x : +1 if x==1 else 0)
#features = ['T','N','Sex','surgery','Bone.metastases','Radiation']
features = ['steatosis','lymphonode','cirrhosis','vascularinvasion','HGB']#
target = 'recur'
#处理数据不平衡
ros = RandomOverSampler(random_state=12, sampling_strategy='auto')
X_ros, y_ros = ros.fit_resample(thyroid_train[features], thyroid_train[target])

XGB = XGBClassifier(random_state=32,max_depth=7,n_estimators=98)
XGB.fit(X_ros, y_ros)
#RF = sklearn.ensemble.RandomForestClassifier(n_estimators=4,criterion='entropy',max_features='log2',max_depth=3,random_state=12)
#RF.fit(X_ros, y_ros)


sp = 0.5
#figure
is_t = (XGB.predict_proba(np.array([[steatosis,lymphonode,cirrhosis,vascularinvasion,HGB]]))[0][1])> sp
prob = (XGB.predict_proba(np.array([[steatosis,lymphonode,cirrhosis,vascularinvasion,HGB]]))[0][1])*1000//1/10

#st.write('is_t:',is_t,'prob is ',prob)
#st.markdown('## is_t:'+' '+str(is_t)+' prob is:'+' '+str(prob))

if is_t:
    result = 'High Risk'
else:
    result = 'Low Risk'
if st.button('Predict'):
    st.markdown('## Risk grouping for relapse:  '+str(result))
    if result == 'Low Risk':
        st.balloons()
    st.markdown('## Probability of relapse:  '+str(prob)+'%')
#st.markdown('## The risk of bone metastases is '+str(prob/0.0078*1000//1/1000)+' times higher than the average risk .')

#排版占行



st.title("")
st.title("")
st.title("")
st.title("")
#st.warning('This is a warning')
#st.error('This is an error')

#st.info('Information of the model: Auc: 0.874 ;Accuracy: 0.851 ;Sensitivity(recall): 0.750 ;Specificity :0.868 ')
#st.success('Affiliation: The First Affiliated Hospital of Nanchang University, Nanchnag university. ')





