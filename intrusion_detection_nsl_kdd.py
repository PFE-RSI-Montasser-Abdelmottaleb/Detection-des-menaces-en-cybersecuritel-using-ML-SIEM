#!/usr/bin/env python
# coding: utf-8

# # 1. IMPORTATION DES LIBRAIRIES



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: '%.3f' % x)
plt.rcParams["figure.figsize"] = (10,6)


# 

# # 2. LECTURE DU JEU DE DONNÉES



file_path = r"D:\pfe_iset\pfe-script-Principale\Detection_Intrusin_nsl_kdd - Copie (2)\data\KDDTest+.txt"  
df = pd.read_csv(file_path)  

# Afficher les premières lignes du DataFrame  
print(df.head())  


# # 2.1 AJUSTER LES COLONNES



columns = (['duration'
,'protocol_type'
,'service'
,'flag'
,'src_bytes'
,'dst_bytes'
,'land'
,'wrong_fragment'
,'urgent'
,'hot'
,'num_failed_logins'
,'logged_in'
,'num_compromised'
,'root_shell'
,'su_attempted'
,'num_root'
,'num_file_creations'
,'num_shells'
,'num_access_files'
,'num_outbound_cmds'
,'is_host_login'
,'is_guest_login'
,'count'
,'srv_count'
,'serror_rate'
,'srv_serror_rate'
,'rerror_rate'
,'srv_rerror_rate'
,'same_srv_rate'
,'diff_srv_rate'
,'srv_diff_host_rate'
,'dst_host_count'
,'dst_host_srv_count'
,'dst_host_same_srv_rate'
,'dst_host_diff_srv_rate'
,'dst_host_same_src_port_rate'
,'dst_host_srv_diff_host_rate'
,'dst_host_serror_rate'
,'dst_host_srv_serror_rate'
,'dst_host_rerror_rate'
,'dst_host_srv_rerror_rate'
,'attack'
,'level'])

df.columns = columns


# We don't have the names of the features from the given dataset so i adjust the columns from : https://www.kaggle.com/code/timgoodfellow/nsl-kdd-explorations



df.head(5)


# # 2.2 INSIGHTS



df.info()


# We have different types of dtypes, we need encoding, doesn't seem like we have null values but we will check



df.describe().T


# There are some outlier values, but we will check if it's too much

# # 3. DATA CLEANING

# # 3.1 NULL VALUES



df.isnull().sum()


# Dataset doesn't contain any null value



#helper function for deeper analysis
def unique_values(df, columns):
    """Prints unique values and their counts for specific columns in the DataFrame."""

    for column_name in columns:
        print(f"Column: {column_name}\n{'-'*30}")
        unique_vals = df[column_name].unique()
        value_counts = df[column_name].value_counts()
        print(f"Unique Values ({len(unique_vals)}): {unique_vals}\n")
        print(f"Value Counts:\n{value_counts}\n{'='*40}\n")




cat_features = df.select_dtypes(include='object').columns
unique_values(df, cat_features)


# Further analysis will be in EDA-VISAULAZTION part about these column's impacts on Attacks

# # 3.2 DUPLICATES



df.duplicated().sum()


# Dataset doesn't contain any duplicated row

# # 3.3 OUTLIERS



df.shape




plt.figure(figsize=(20, 40))
df.plot(kind='box', subplots=True, layout=(8, 5), figsize=(20, 40))
plt.show()


# There is no too much outlier to misslead the model so i will not drop the outliers

# # 3.4 CLASSIFY ATTACK OR NOT



attack_n = []
for i in df.attack :
  if i == 'normal':
    attack_n.append("normal")
  else:
    attack_n.append("attack")
df['attack'] = attack_n 




df['attack'].unique()


# # 4. EDA - VISUALIZATIONS



df.hist(bins=43,figsize=(20,30));


# General visualization in order to get insights

# # 4.1 Protocol Type



plt.figure(figsize=(16,4))
sns.countplot(x='attack',data=df,hue='protocol_type')
plt.xticks(rotation=45)
plt.title('Attack Counts over Protocol Types',fontdict={'fontsize':16})
plt.show()




# So we can see that most of the attacks are from tcp, then udp, and least attack comes from icmp




df["protocol_type"].value_counts(normalize=True)


# # 4.2 Service used general



import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 8))  # Adjusted figure size
ax = sns.countplot(x='service', data=df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")  # Rotated labels
plt.xlabel('Service')
plt.ylabel('Count')
plt.title('Count of Services')
plt.grid(True)
plt.show()




# Services most used in general follows as, http,private,domain_u,smtp, ftp,other..


# # 4.3 Service used effect on attacks



import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 8))  # Adjusted figure size
ax = sns.countplot(x='service', hue='attack', data=df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")  # Rotated labels
plt.xlabel('Service')
plt.ylabel('Count')
plt.title('Distribution of Attacks by Service')
plt.legend(title='Attack Type')
plt.grid(True)
plt.show()




#we can see that private attacks is most common service 


# # 4.4 Kernel Density Estimate (KDE) Plot of Duration by Flag



plt.figure(figsize=(12, 8))
sns.displot(
    data=df,
    x="duration",
    hue="flag",
    kind="kde",
    height=6,
    multiple="fill",
    clip=(0, None),
    palette="ch:rot=-.25,hue=1,light=.75",
)
plt.title('Kernel Density Estimate (KDE) Plot of Duration by Flag')
plt.grid(True)
plt.show()


# # 4.5 Distribution of Attack Types by Guest Login



plt.figure(figsize=(10, 6))
sns.countplot(x='is_guest_login', hue='attack', data=df, palette='Set2')
plt.xlabel('Is Guest Login')
plt.ylabel('Count')
plt.title('Distribution of Attack Types by Guest Login')
plt.legend(title='Attack Type')
plt.grid(True)
plt.show()


# we can clearly say that attacks are comes when guest is not login

# # 5. PREPROCESSING

# # 5.1 ENCODING



cat_features = df.select_dtypes(include='object').columns
cat_features




from sklearn import preprocessing
le=preprocessing.LabelEncoder()
clm=['protocol_type', 'service', 'flag', 'attack']
for x in clm:
    df[x]=le.fit_transform(df[x])


# # 5.2 TRAIN-TEST-SPLIT



from sklearn.model_selection import train_test_split

X = df.drop(["attack"], axis=1)
y = df["attack"]

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.1,random_state=43) 




train_index = X_train.columns
train_index


# # 5.3 Feature Engineering



from sklearn.feature_selection import mutual_info_classif
mutual_info = mutual_info_classif(X_train, y_train)
mutual_info = pd.Series(mutual_info)
mutual_info.index = train_index
mutual_info.sort_values(ascending=False)




mutual_info.sort_values(ascending=False).plot.bar(figsize=(20, 5));


# # 5.4 Feature Selection



from sklearn.feature_selection import SelectKBest
Select_features = SelectKBest(mutual_info_classif, k=30)
Select_features.fit(X_train, y_train)
train_index[Select_features.get_support()]




columns=['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
       'dst_bytes', 'wrong_fragment', 'hot', 'logged_in', 'num_compromised',
       'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate']

#We will continue our model with top 15 features, because dataset is big enough

X_train=X_train[columns]
X_test=X_test[columns]


# # 5.5 Scaling



from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) # we use only transform in order to prevent data leakage


# # 6. MODEL BUILD



from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib 




XGBoost_model = XGBClassifier(random_state = 42)
Logistic_model = LogisticRegression(random_state=42)




XGBoost = XGBoost_model.fit(X_train,y_train)




Logistic = Logistic_model.fit(X_train,y_train)




from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score




#it's a helper function in order to evaluate our model if it's overfit or underfit.
def eval_metric(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)

    print("Test_Set")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print()
    print("Train_Set")
    print(confusion_matrix(y_train, y_train_pred))
    print(classification_report(y_train, y_train_pred))




eval_metric(Logistic_model, X_train, y_train, X_test, y_test)




eval_metric(XGBoost_model, X_train, y_train, X_test, y_test)


# So we can see that ensemble methods such as xgboost,adaboost,gradientboosts has more accurace scores over logistic regression in bigger datasets.
# 
# It doesn't neccessary but we will do hyperparameter tuning in order to fit the model with best parameters, i would like to remember that xgboost has cross-validation has itself

# # 6.1 HYPERPARAMETER TUNING



param_grid = {
    "n_estimators": [50,64,100,128],
    "max_depth": [2, 3, 4,5,6],
    "learning_rate": [0.01,0,0.03, 0.05, 0.1],
    "subsample": [0.5, 0.8],
    "colsample_bytree": [0.5, 0.8]
}




from sklearn.model_selection import GridSearchCV

XGB_model = XGBClassifier(random_state=42) #initialize the model

XGB_grid_model = GridSearchCV(XGB_model,
                        param_grid,
                        scoring="f1",
                        n_jobs=-1,
                        return_train_score=True).fit(X_train, y_train)




XGB_grid_model.best_score_




XGB_grid_model.best_params_


# # 6.2 FINAL MODEL



XGB_model = XGBClassifier(
    colsample_bytree=0.5,
    learning_rate=0.1,
    max_depth=6,
    n_estimators=128,
    subsample=0.8
)

# Fit the classifier to your data
XGB_model.fit(X_train, y_train)


# # 6.3 EVALUATION



y_pred = XGB_model.predict(X_test)
y_pred_proba = XGB_model.predict_proba(X_test)

xgb_f1 = f1_score(y_test, y_pred)
xgb_recall = recall_score(y_test, y_pred)
xgb_auc = roc_auc_score(y_test, y_pred_proba[:,1])




xgb_auc




from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_estimator(XGB_model, X_test, y_test);




eval_metric(XGB_model, X_train, y_train, X_test, y_test)


# # 7. FEATURE IMPORTANCE



model = XGB_model
model.feature_importances_

feats = pd.DataFrame(index=X[columns].columns, data= model.feature_importances_, columns=['XGB_importance'])
ada_imp_feats = feats.sort_values("XGB_importance", ascending = False)
ada_imp_feats




y_pred




y_pred_string = le.inverse_transform(y_pred)
y_pred_string




# Create the countplot
plt.figure(figsize=(10, 6))
sns.countplot(x=y_pred_string, palette="pastel")

# Add labels and title
plt.xlabel("Attack Type")
plt.ylabel("Count")
plt.title("Distribution of Attack Types")

# Show the plot
plt.show()




# Enregistre le modèle entraîné  
joblib.dump(XGB_model, 'xgb_model.joblib')  
print("Le modèle a été sauvegardé sous le nom 'xgb_model.joblib'.")  




# Étape 2 : Charger et utiliser le modèle  

# Charger le modèle sauvegardé  
loaded_model = joblib.load('xgb_model.joblib')  

# Utiliser le modèle chargé pour faire des prédictions  
predictions = loaded_model.predict(X_test)  

# Évaluer les prédictions  
accuracy = accuracy_score(y_test, predictions)  
print("Précision des prédictions :", accuracy)  
print("Prédictions :", predictions)  
