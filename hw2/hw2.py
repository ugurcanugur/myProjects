#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 18:35:08 2022

@author: ugurcanugur
"""

import numpy as np
import matplotlib.pyplot as plt
import re




with open("Medium_DS_Articles_2021_Dataset.csv") as t:
    text=[]
    uniqtxt=[]
    a=t.readlines()
    fline_skip=0
    
    for i in a:
        if fline_skip!=0:
            b=i.split(',')
            
            text. append(b)
            if b[5] not in uniqtxt:
                uniqtxt.append(b[5])
        else:
            fline_skip+=1
    

arr = np.array(text)   
"""  
print("Number of rows:",len(text)," columns:",len(text[1]))
print("Number of unique Tags:",len(uniqtxt))
"""
print("Number of rows: {}".format(arr.shape[0])," columns: {}".format(arr.shape[1]))
print("Number of unique Tags:",len(uniqtxt))

#########
dumlist=[]
d={}
for i in range( len(uniqtxt)):
    d.update({uniqtxt[i]:0})
    dumlist.append(0)
    

for i in text:
    count=0
    for j in uniqtxt:
        if j==i[5]:
             dumlist[count]+=1
             d.update({j:dumlist[count]})
        count+=1
newdic=sorted(d.items(), key=lambda x: x[1])
rdic=(list(reversed(newdic)))
for i in rdic:
    print(i[0],i[1])
###########
z=arr[:,6]

def validate_date(x):


    newx=re.sub(r'-','',x)
    if x ==newx:
        return False
    newx=re.sub(r':','',x)
    if x != newx :
        return False
    else:
              
        x=x.split('-')
        
        if len(x[0])==4 and len(x[1])==2 and len(x[2])==3:
            return True
        else :
            return False
       
#validate_date(x=arr[151,6])


def validate_date_format(z):
  
  for x in z :
    if validate_date(x) == False:
      return False 
  else:
    return True


look=validate_date_format(z)
print(look)

def clean_date(z):
  datelist=[]        
  c=0
  clms=[]
  darr= arr
  for x in z :
      
      if validate_date(x)== True:
          c+=1
          
      else:
          
          unp=re.sub(r'[^\w\s]','',x)
          unp=unp.split(' ')
          
          if len(unp)==1:
            dumdate=unp
            newdate=[dumdate[0][0:4],dumdate[0][4:6],dumdate[0][6:9]]
            datelist.append(newdate)
          elif len(unp)==3:
            datelist.append(unp)
            
          clms.append(c)
          c+=1

  darr =np.delete(darr, clms, axis=0) 
     
  return darr,datelist
    
newdata,datelist=clean_date(z)
print(validate_date_format(newdata[:,6]))

newdataQ2=newdata

uniqdate=[]
date_count=[]
monthlist=[]

for i in datelist:
    monthlist.append(i[0]+","+i[1])




for i in range(len(monthlist)):
    if monthlist[i] not in uniqdate:
        uniqdate.append(monthlist[i])

for i in range(len(uniqdate)):
    date_count.append(1)


for i in range(len(monthlist)):
    if monthlist[i]  in uniqdate:
        index=uniqdate.index(monthlist[i])
        date_count[index]+=1
newuniqdate=[]
for i in uniqdate:
    newuniqdate.append([i])
    





fig = plt.figure(figsize = (len(date_count), len(uniqdate)))

plt.bar(uniqdate, date_count, color ='blue',
        width = 0.8)
 
plt.xlabel("dates")
plt.ylabel("numbers")
plt.title("Q3")
plt.show()


read_time=[]

for i in newdata:
    a=i[4].split('-')
    read_time.append(float(a[0]))
    
read_timearr=np.array(read_time)

print("After cleaning, Number of rows: {}".format(newdataQ2.shape[0])," columns: {}".format(newdataQ2.shape[1]))



print("Maximum reading time: ",np.max(read_timearr))
print("Minimum reading time: ",np.min(read_timearr))
print("Average reading time: ",np.mean(read_timearr))






Interaction=[]

for i in range(len(newdata)):
    if float(newdata[i][2])==0 and float(newdata[i][3])==0:
        Interaction.append([0])
        
    else:
        Interaction.append([1])

        
newdata=np.append(newdata, Interaction ,axis=1)
sumprob=[]
counter=0
temprdic=[]
for i in rdic:
    if counter==5:
      counter+=1
      continue
    else:
        temprdic.append(i)
        counter+=1

rdic=temprdic
uniq_txt=[]
for i in temprdic:
    uniq_txt.append(i[0])
   


for a in uniq_txt :
    sumprob.append(0)
   
for i in newdata:
    endex=uniq_txt.index(i[5])
    if float(i[7])==1:
        sumprob[endex]+=1
prob=[]

for i in  range(len(rdic)):
    prob.append( sumprob[i] / rdic[i][1])
for i in range (len (prob)):
    print( rdic[i][0],"---",  prob[i])
    








