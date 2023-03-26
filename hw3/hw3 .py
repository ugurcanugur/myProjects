
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join

#import geopandas as gpd



df = pd.read_csv("/Users/ugurcanugur/Desktop/ders/cs 210/hw3/Job_Satisfaction_Survey.csv")
olddf=df
def isNaN(num):
    return num != num

def remowingColums(df):
   
    sp=df.shape
   
    print("Initially, there are", sp[0], "rows and ",sp[1]," columns")
   
    delList=[]
    for j in range(sp[0]):
       
        counter=0
        for i in df.iloc[j]:
           
            if isNaN(i):
                counter+=1
        if counter >= (sp[1]/4):
            per=round(int(counter)/sp[1]*100)
            delList.append(j)
           # print( per," % of the values are missing in",j)
           
           
    df= df.drop(delList)
    df.reset_index(inplace=True)
    sp=df.shape
    print("After removing columns with high missing value percentage, there are",sp[0],"columns ")
    return(df)
#df=remowingColums(df)

##############################

def addSkillNumRow(df):
   
   
    MLSkillsSelectnum=[]
    MLTechniquesSelectnum=[]
   
    for i in df["MLSkillsSelect"]:
        if isNaN(i):
            MLSkillsSelectnum.append(0)
        else:
            MLSkillsSelectnum.append(len(str(i).split(",")))
           
           
       
    for i in df["MLTechniquesSelect"]:
        if isNaN(i):
            MLTechniquesSelectnum.append(0)
        else:
            MLTechniquesSelectnum.append(len(str(i).split(",")))
       
    df.insert(9, "MLSkillsSelect_Amount",list(MLSkillsSelectnum) , True)
    df.insert(11, "MLTechniquesSelect_Amount", list(MLTechniquesSelectnum) , True)
    return df
   
addSkillNumRow(df)

def writeStats(df):
    return print(df[["MLSkillsSelect_Amount","MLTechniquesSelect_Amount"] ]
                 .describe(include="all"))
   
writeStats(df)
##############################

def ageFilter(df):
    df = df[df['Age'].between(18, 64)]
    df.reset_index(inplace=True)
    print("Number of rows after filtering Age column is",df.shape[0])
    return df
       
df=ageFilter(df)
   
def incomeDollar(df):
    dollarlist=[]
    incomelist=df["Income"]
    IncomeCurrencylist=df["IncomeCurrency"]
    for i in range(df.shape[0]):
       if IncomeCurrencylist[i] =="dollar" :
           
           dollarlist.append(incomelist[i])
           
       else:
            dolars=incomelist[i]* (104/100)
            dollarlist.append(dolars)
    df.insert(24, "Income_Dollar",list(dollarlist) , True)
    return df
   
incomeDollar(df)

def ageCluster(df):
    agelist=[]
    for i in df['Age']:
       
        if isNaN(i):
            agelist.append('0')
        else:
            if int(i)>=18 and int(i)<30 :
                agelist.append('[18-30)')
            elif int(i)>=30 and int(i)<45 :
                agelist.append('[30-45)')
            elif int(i)>=45 and int(i)<55 :
                agelist.append('[45-55)')
            elif int(i)>=55 and int(i)<65 :
                agelist.append('[55-65)')
            else:
                agelist.append('0')
       
           
    df.insert(4, "Age_Group",list(agelist) , True)
    return df
ageCluster(df)



def plot_Edu_Age_relation(df):
    fageQuantitiy=[]
    uniqdegreelist=[]
    
    
    for i in df['FormalEducation']:
        
        if i not in uniqdegreelist and isNaN(i)==False:
            uniqdegreelist.append(i)
    
    ages=['[18-30)','[30-45)','[45-55)','[55-65)']
    
    totalincome=[0,0,0,0]
    ageq=[0,0,0,0]
    for i in uniqdegreelist :
        
        tempdf=df[df['FormalEducation'].between(i,i)]
        tempdf.reset_index(inplace=True, drop=True)
        c=0
        dollarlist =tempdf['Income_Dollar']
        for j in tempdf['Age_Group']:
                if j!= '0' :
                    idx=ages.index(j)
                    totalincome[idx]+=round(dollarlist[c])
                    ageq[idx]+=1
                c+=1
        avarageincome=[0,0,0,0]
        for a in range (len(totalincome)):       
            avarageincome[a]= totalincome[a]/ageq[a] 
        fageQuantitiy.append(avarageincome)
        totalincome=[0,0,0,0]
        ageq=[0,0,0,0]
       
    plotdata = pd.DataFrame({str(ages[i]) : fageQuantitiy[i] for i in range(len(ages)) },
                            index=uniqdegreelist)
    plotdata.plot(kind="bar")
    plt.title("Mince Pie Consumption Study")
    plt.xlabel("Family Member")
    plt.ylabel("Pies Consumed")
    return df , plt.show()
    

   
            
plot_Edu_Age_relation(df)        
        
    
####################!!!!!!
def corrolationHeat(df):
    corrdf=df.filter(['Age',
                      'JobSatisfaction',
                      'Income_Dollar',
                      "MLSkillsSelect_Amount",
                      "MLTechniquesSelect_Amount"])
                     
    
    corr = corrdf.corr()
    
    sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        annot=True)
    plt.title('Correlation of Features')
    plt.show()
    
    
corrolationHeat(df)
#######################

def Joy_Satisfaction_Around_The_World(df):
    avg_df = df[["Country","JobSatisfaction"]].groupby("Country").mean().reset_index()
    idx=[]
    uniqCountry=[]
    numcountry=0
    c=0
    for i in df['Country']:
        if i not in uniqCountry:
            uniqCountry.append(i)
    dic={uniqCountry[j]:0 for j in range(len(uniqCountry))}
    numdic={uniqCountry[j]:0 for j in range(len(uniqCountry))}
    for i in df['Country']:
        if i  in uniqCountry:
            dic[i] = dic.get(i, 0) + df.iloc[c]['JobSatisfaction']
            numdic[i] = numdic.get(i, 0) +1
        c+=1
    for i in uniqCountry:
        if  numdic.get(i, 0) != 0 :
            dic[i] = dic.get(i, 0) / numdic.get(i, 0)
        else:
            dic[i]=0
    locdf = pd.DataFrame.from_dict(dic,orient='index')
    locdf.columns = ["satisfaction"]
    
    return dic,locdf
    
dic,locdf=Joy_Satisfaction_Around_The_World(df)


def plot_world(locdf):
    title = 'job satisfaction araound the world'
    col = 'case_growth_rate'
    cmap = 'OrRd'    
    fig, ax = plt.subplots(1, figsize=(20, 5))
    
    ax.axis('off')
    df.plot(column=col, ax=ax, edgecolor='0.8', linewidth=1, cmap=cmap)
    ax.set_title(title, fontdict={'fontsize': '25', 'fontweight': '3'})
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    cities = gpd.read_file(gpd.datasets.get_path('naturalearth_cities'))
    world.head()
    world['gdp_per_cap'] = locdf

    world.plot(column='gdp_per_cap');
plot_world(locdf)
    
    




