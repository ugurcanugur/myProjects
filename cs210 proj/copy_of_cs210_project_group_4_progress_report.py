
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats


df = pd.read_csv('Placement_Data_Full_Class.csv')
df.head()

"""sl_no acts as an index, and we don't need it for modeling. So let's delete it."""

df.drop(columns=['sl_no'], inplace=True)

print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

"""Currently we have 215 rows and 14 columns in the dataset, and here are the data types:"""

df.info()

"""We can see that the data types are either object or float64.  
Let's examine the categorical variables by viewing their distribution:
"""

categorical_list = list(df.select_dtypes(include=['object']).columns)

print("Categorical variables:", categorical_list)

#For each categorical variable, plot the counts in a bar plot
for variable in categorical_list:

  column = df[variable]
  values = column.value_counts()

  #Print the name and value counts for the current categorical variable
  print("{}:\n\n{}\n".format(variable, values))
  
  #Plot
  plt.figure(figsize = (9,3))
  plt.bar(values.index, values)
  plt.xticks(values.index, values.index.values)
  plt.ylabel("Count")
  plt.title(variable)
  plt.show()
  
  print("\n##################################################################\n")

"""Let's also view the distributions of the numerical variables:"""

numerical_list = list(df.select_dtypes(include=['float64']).columns)

print("Numerical variables:", numerical_list)

for variable in numerical_list:

  print("{}:\n".format(variable))

  #Plot a histogram for the variable
  plt.figure(figsize = (9,3))
  plt.hist(df[variable], bins = 40)
  plt.xlabel(variable)
  plt.ylabel("Count")
  plt.title("{} distribution".format(variable))
  plt.show()
  
  print("\n##################################################################\n")

print("{}:\n".format(variable))

  #Plot a histogram for the variable
plt.figure(figsize = (9,3))
plt.hist(df[variable], bins = 10)
plt.xlabel('salary')
plt.ylabel("Count")
plt.title("{} distribution".format('salary'))
plt.show()

print("\n##################################################################\n")

plt.figure(figsize=(20,12))

colors = ['red', 'green', 'blue', 'pink', 'yellow', 'black']

for n in range(len(numerical_list)):
  plt.subplot(2,3,n+1)
  sns.histplot(df[numerical_list[n]], color = colors[n], kde = True)

import numpy
tempdf=df['salary']
tempdf.dropna()
data = numpy.asarray(tempdf)
print(data)
m1 = numpy.average(data)
print(m1)
lambdaEst = 1/m1

print("Lambda estimate: ", lambdaEst)

tempdf=df['etest_p']
tempdf.dropna()
data = numpy.asarray(tempdf)

column = df[variable]
values= df['etest_p'].value_counts()

print(values,values.shape)
m1 = numpy.average(data)
print(m1)
lambdaEst = 1/m1

print("Lambda estimate: ", lambdaEst)
plt.figure(figsize=(20,12))

colors = ['red', 'green', 'blue', 'pink', 'yellow', 'black']

for n in range(len(numerical_list)):
  plt.subplot(2,3,n+1)
  sns.boxenplot(x=df['status'], y=df[numerical_list[n]], color=colors[n], scale="linear", data=df)


df.isna().sum()

print("{}% of the salary column is missing.".format((df['salary'].isnull().sum() / (len(df)) * 100).round(2)))

"""The salary is NaN for the students who were not placed, because they are not working. This means that 31.16% of students were not placed."""

df[['status', 'salary']][np.isnan(df.salary)]

"""We are going to try the predict the placements, so let's inspect the relationship between the other variables with placements in more detail.  
Let's start with the degree percentage's impact on the placement:
"""

plt.figure(figsize=(20,5))

plt.subplot(1,3,1)
sns.boxenplot(x=df['status'], y=df['degree_p'], color="b", scale="linear", data=df)

plt.subplot(1,3,2)
sns.swarmplot(x="status", y="degree_p", data=df)

plt.subplot(1,3,3)
sns.scatterplot(x="hsc_p", y="degree_p", data=df, hue="status")

plt.show()

"""When we look at the plots above, we can see that degree percentage is important to get placed. It can be said that the higher the 'degree_p' value, the higher the probability of getting placed, which is as expected.

Let's convert some categorical columns to numeric. This will make plots easier to work with.
"""

df.status = df.status.map({'Placed': 1, 'Not Placed': 0})
df.gender = df.gender.map({'M': 0, 'F': 1})

"""Let's look at the effect of gender in placements."""

gender_plot = sns.catplot(y='status', x='gender', data=df, height=5, aspect=2, kind='bar')
gender_plot.set_xticklabels(['Male', 'Female'], size=15)
gender_plot.fig.suptitle('Proportion of placements by gender', size=20, y=1.05)
plt.show()

"""We can see that male students had a slightly higher chance of getting placed. Let's perform a hypothesis test to see if there is bias on the gender."""

#status: 0 represents Not Placed, 1 represents Placed
#gender: 0 represents Male, 1 represents Female
gender_crosstab = pd.crosstab(df['gender'], df['status'], margins = False) 
gender_crosstab

#This is a chi square test to find out if placements are affected by gender
H0 = "Gender does not have an effect on placement" #Null hypothesis
Ha = "Placement is affected by gender" #Alternative hypothesis

chi, p_value, dof, expected = stats.chi2_contingency(gender_crosstab)

significance = 0.05 #Significance level is 5%

if p_value < significance:
    print("{} because the p_value is {} which is less than {}".format(Ha, p_value.round(3), significance))
else:
    print("{} because the p_value is {} which is greater than {}".format(H0, p_value.round(3), significance))

"""Now, let's look at which specialization in higher secondary education (hsc_s) is more favored for the placements:"""

#Assign 0 to Commerce, 1 to Science, 2 to Arts
df.hsc_s = df.hsc_s.map({'Commerce': 0, 'Science': 1, 'Arts': 2})

fig = plt.figure(figsize=(12, 12))

plt.subplot(2,1,1)
hsc_s_plot = sns.countplot(x=df.hsc_s)
hsc_s_plot.set_title('Effect of specialization on placement', fontsize=20, y=1.05)
hsc_s_plot.set_xticklabels(['Commerce', 'Science', 'Arts'], size=15)
hsc_s_plot.set(xlabel=None)

plt.subplot(2,1,2)
hsc_s_status = sns.pointplot(x=df.hsc_s, y='status', data=df, ci=95, join=False)
hsc_s_status.set_xticklabels([])

plt.show()


plt.figure(figsize=(10, 8))

sns.heatmap(df.corr(),annot=True)
plt.show()


from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

string_list = list(df.select_dtypes(include=['object']).columns)

olddf=df
df = df.drop(string_list, axis=1)
df=df.dropna()

y= df["status"]
X= df.drop(["status"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.50, random_state=42)

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

#Decision Tree Training
model_dt = tree.DecisionTreeClassifier(random_state=42) #Create decision tree classifier object
model_dt.fit(X_train, y_train) #train the classifier using the training data


model_rf = RandomForestClassifier(n_estimators=5, random_state=42)
model_rf.fit(X_train, y_train)



from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve

#Decision Tree Testing
dt_predictions_val = model_dt.predict(X_val)
dt_acc_val = accuracy_score(y_val, dt_predictions_val)

rf_predictions_val = model_rf.predict(X_val)
rf_acc_val = accuracy_score(y_val, rf_predictions_val)

print("Decision Tree Validation Accuracy:"+str(dt_acc_val))
print("Random Forest Validation Accuracy:"+str(rf_acc_val))










