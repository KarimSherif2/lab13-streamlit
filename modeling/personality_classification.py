# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

df = pd.read_csv('modeling/16P.csv', encoding='latin1')

z = df.columns

df.head()

df.drop('Response Id',axis=1,inplace=True)

questions = list(df.columns)

print("Questions of the test:")
i = 1
for x in df.columns:
    print(f'\t{i}-{x}')
    i = i + 1



df = df.drop_duplicates()

y = df["Personality"].copy()
df.drop("Personality",axis=1,inplace=True)

qq = ['Q'+str(i) for i in range(1,len(df.columns)+1)]
df.columns = qq

df

questions_map = {qq[i]: questions[i] for i in range(len(qq))}


le = LabelEncoder()
y = le.fit_transform(y)



x_train,x_test,y_train,y_test = train_test_split(df,y,test_size=0.2)

sv = SVC(C=1, gamma=0.01, kernel='rbf')

sv.fit(x_train,y_train)


sv.score(x_train,y_train)
sv.score(x_test,y_test)


import joblib

joblib.dump(sv, 'svc_model.pkl')

joblib.dump(le, 'label_encoder.pkl')


columns = df.columns.tolist()

columns_filtered = [col for col in columns if 'Personality' not in col]

joblib.dump(columns_filtered, 'questions.pkl')
