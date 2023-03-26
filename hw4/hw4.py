
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import warnings
warnings.filterwarnings('ignore')
def isNaN(num):
   
    return num != num

filename = "creditcard_fraud.csv"
df = pd.read_csv('/Users/ugurcanugur/Desktop/cs 210/hw4/creditcard_fraud.csv', encoding='utf-8')

df.head()
def plotpie(df):   
    add=0
    for i in df['Class']:
        if i==1:
            add+=1
    fig, ax = plt.subplots(figsize =(10, 7))  
    plt.pie([add,df.shape[0]], labels = ['Fraud','Non-fraud'] ,autopct='%1.1f%%' )
    ax.set_title("Customizing pie chart")
    plt.show()

########plotpie(df)
#Q1


from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

string_list = list(df.select_dtypes(include=['object']).columns)


df = df.drop(string_list, axis=1)
#droping time collumn
df.drop(columns=['Time'], inplace=True)

#looking for class ml 
y= df["Class"]
X= df.drop(["Class"], axis=1)
y_true=y
# 60% for training and 40% for testing-validation
X_train, X_remaining, y_train, y_remaining = train_test_split(X, y, test_size=0.40, random_state=0)
# 20% validation, 20% test
X_test, X_val, y_test, y_val = train_test_split(X_remaining, y_remaining, test_size=0.50, random_state=0)



##Q2


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
#########random forest
######### number of estimators=50
model_rf = RandomForestClassifier(n_estimators=50, random_state=0 )

classifier_rf= model_rf.fit(X_train, y_train)

#predict for test
y_pred_rf = model_rf.predict(X_test)


##########GradientBoostingClassifier 
###########learning_rate=0.1
gradient_booster = GradientBoostingClassifier(learning_rate=0.1,random_state=0)


classifier_gb= gradient_booster.fit(X_train, y_train)
#predict for test
y_pred_gb = gradient_booster.predict(X_test)

############Q3





from sklearn.metrics import confusion_matrix

confusion_matrix_rf = confusion_matrix(y_test, y_pred_rf)

confusion_matrix_gb = confusion_matrix(y_test, y_pred_gb)

######ploting
from sklearn.metrics import plot_confusion_matrix

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))


disp_rf =plot_confusion_matrix(classifier_rf,X, y_true,cmap=plt.cm.Blues, ax=ax1)

disp_gb =plot_confusion_matrix(classifier_gb,X, y_true,cmap=plt.cm.Greens, ax=ax2)


'''
disp_rf.ax1.set_title('Confusion Matrix (Random Forest)')
disp_gb.ax2.set_title('Confusion Matrix (Gradient Boosting)')
'''

plt.show()
#####Q4





from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, roc_curve, auc, precision_recall_curve
from sklearn import metrics
from sklearn.datasets import make_classification


fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True,figsize=(15, 7))
ax1, ax2, ax3, ax4 = axes.flatten()

RocCurveDisplay.from_estimator(
   model_rf, X_test, y_test, ax = ax1,color='green')

RocCurveDisplay.from_estimator(
   gradient_booster, X_test, y_test, ax = ax2)


#precision_recall_curve for rf

y_score_rf = classifier_rf.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_score_rf)
ax3.plot(recall, precision, color='purple')
#precision_recall_curve for gb

y_score_gb = classifier_gb.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_score_gb)
ax4.plot(recall, precision, color='brown')


########Q5
n_estimators=[50,100,300,500]
max_features=[5,7,10,25]
from sklearn.metrics import average_precision_score

pred_scors=[]
print('random forest:')
for i in n_estimators:
    
  #random forest
  model_rf = RandomForestClassifier(n_estimators=i, random_state=0 )

  classifier_rf= model_rf.fit(X_train, y_train)
  y_score_rf = classifier_rf.predict_proba(X_test)[:, 1]
  average_precision = average_precision_score(y_test, y_score_rf)
  pred_scors.append(average_precision)
  print('n_estimators',i,' AUPRC score:',average_precision)

dic_rf={n_estimators[i] : pred_scors[i] for i in range(len(n_estimators))}
sorteddic=sortdic=dict(sorted(dic_rf.items(), key=lambda item: item[1]))
best_estimator=list(sorteddic)[-1]
print('Best n_estimators for random forest :',best_estimator)


max_features=[5,7,10,25]
pred_scors_max_features=[]
for i in max_features:

  model_rf = RandomForestClassifier(n_estimators=best_estimator, random_state=0,max_features=i )

  classifier_rf= model_rf.fit(X_train, y_train)
  y_score_rf = classifier_rf.predict_proba(X_test)[:, 1]
  average_precision = average_precision_score(y_test, y_score_rf)
  pred_scors_max_features.append(average_precision)
  print('n_estimators',best_estimator,'max_features',i,' AUPRC score:',average_precision)

dic_max_rf={max_features[i] : pred_scors_max_features[i] for i in range(len(max_features))}

sorteddic_max=sortdic=dict(sorted(dic_max_rf.items(), key=lambda item: item[1]))

best_estimator_max=list(sorteddic_max)[-1]
print('for random forest, best n_estimators=',best_estimator,'and the best max_features=',best_estimator_max )


n_estimators=[50,100,300,500]

estimator_gb=[]
print('gradient_booster:')

for i in n_estimators:
    
  #gradient_booster
  gradient_booster = GradientBoostingClassifier(learning_rate=0.1,random_state=0,n_estimators=i)

  classifier_gb= gradient_booster.fit(X_train, y_train)

  y_score_gb = classifier_gb.predict_proba(X_test)[:, 1]
  average_precision = average_precision_score(y_test, y_score_gb)
  estimator_gb.append(average_precision)
  print('n_estimators',i,' AUPRC score:',average_precision)


dic_gb={n_estimators[i] : estimator_gb[i] for i in range(len(n_estimators))}
sorteddic_gb=sortdic=dict(sorted(dic_gb.items(), key=lambda item: item[1]))
best_estimator_gb=list(sorteddic_gb)[-1]
print('Best n_estimators for random forest :',best_estimator_gb)


#####

max_features=[5,7,10,25]
pred_scors_max_features=[]
for i in max_features:
  gradient_booster = GradientBoostingClassifier(learning_rate=0.1,random_state=0,n_estimators=best_estimator_gb,max_features=i)

  classifier_gb= gradient_booster.fit(X_train, y_train)

  y_score_gb = classifier_gb.predict_proba(X_test)[:, 1]
  average_precision = average_precision_score(y_test, y_score_gb)
  pred_scors_max_features.append(average_precision)
  print('n_estimators',best_estimator,'max_features',i,' AUPRC score:',average_precision)

dic_max_features_gb={max_features[j] : pred_scors_max_features[j] for j in range(len(max_features))}
sorteddic_max=sortdic=dict(sorted(dic_max_features_gb.items(), key=lambda item: item[1]))
best_estimator_max=list(sorteddic_max)[-1]
print('for random forest, best n_estimators=',best_estimator,'and the best max_features=',best_estimator_max )



######Q6
#random forest
dic_rf
dic_max_rf
#g b
dic_gb
dic_max_features_gb


x_rf_estimator=list(dic_rf.keys())
y_rf_estimator=list(dic_rf.values())

x_gb_estimator=list(dic_gb.keys())
y_gb_estimator=list(dic_gb.values())


plt.plot(x_rf_estimator, y_rf_estimator,color='blue',label='da',
         linestyle='dashed', linewidth = 3, marker='o', markerfacecolor='blue',markersize=10)

plt.plot(x_gb_estimator, y_gb_estimator,color='red',
         linestyle='dashed', linewidth = 3, marker='o', markerfacecolor='red',markersize=10)

plt.ylim(0,1)

plt.grid(True)
plt.yticks= np.around(np.linspace(0.7,1),decimals=2)



x_rf_depht=list(dic_max_rf.keys())
y_rf_debth=list(dic_max_rf.values())

#plt.plot(x_rf_depht, y_rf_debth)



plt.show()



















