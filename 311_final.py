#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 18:16:35 2023

@author: ugurcanugur
"""
#ugurcan ugur 26448
from gurobipy import GRB,Model,quicksum 


def Output(m):  
    # Print the result
    status_code = {1:'LOADED', 2:'OPTIMAL', 3:'INFEASIBLE', 4:'INF_OR_UNBD', 5:'UNBOUNDED'} #this is how a 'dictionary' 
                                                                                            #is defined in Python
    status = m.status
    
    print('The optimization status is ' + status_code[status])
    if status == 2:    
        # Retrieve variables value
        print('Optimal solution:')
        for v in m.getVars():
            print(str(v.varName) + " = " + str(v.x))    
        print('Optimal objective value: ' + str(m.objVal) + "\n")
        
        
      
        
def OptModel(d,t,C,f):
    # Create the model
    model = Model('FINAL')
    
    # Set parameters
    model.setParam('OutputFlag',True)
    
    n = len(d)#j
    m = len(t)#i
    
   
    x= model.addVars(m,n, lb=0,  vtype=GRB.CONTINUOUS,name="production ")
    y= model.addVars(m,  vtype=GRB.BINARY,name="fixed cost ")
    a= model.addVars(m,n,  vtype=GRB.BINARY,name="  ")
    
   
    
   
    model.addConstrs(quicksum(x[i,j] for i in range(m)) ==d[j]  for j in range(n))
    
    #Q2
    model.addConstrs(quicksum(x[i,j] for j in range(n)) <=C[i]  for i in range(m))
    
    #Q3
    BIGM=10000000000
    #model.addConstrs(x[i,j]  <=y[i,j]*BIGM for j in range(n) for i in range(m))
    model.addConstrs(quicksum(x[i,j] for j in range(n)) <=y[i]*BIGM  for i in range(m))
    
    #Q4
    model.addConstrs(x[i,j]  <=a[i,j]*BIGM for j in range(n) for i in range(m))
    model.addConstrs(quicksum(a[i,j]for i in range(m)) <=1 for j in range(n) )
    
   
   
    model.setObjective(quicksum(t[i][j]* x[i,j]  for i in range(m) for j in range(n) ), GRB.MINIMIZE)
    #for Q4 answer
    #model.setObjective(quicksum(t[i][j]* x[i,j]+y[i]*f[i]  for j in range(n) for i in range(m) ), GRB.MINIMIZE)
    # Optimize the model
    model.optimize()
    
        
    Output(model)


 
f=[4000,3000,5000]
C=[500,1000,700]
d =[200,100,400,600,300]
t = [[9,3,5,4,2],
     [4,2,5,9,6],
     [6,4,3,7,8]]
OptModel(d ,t,C,f)   



"""
Q1
production [0,0] = 0.0
production [0,1] = 0.0
production [0,2] = 0.0
production [0,3] = 600.0
production [0,4] = 300.0
production [1,0] = 200.0
production [1,1] = 100.0
production [1,2] = 0.0
production [1,3] = 0.0
production [1,4] = 0.0
production [2,0] = 0.0
production [2,1] = 0.0
production [2,2] = 400.0
production [2,3] = 0.0
production [2,4] = 0.0
Optimal objective value: 5200.0
"""

"""
Q2
Optimal solution:
production [0,0] = 0.0
production [0,1] = 0.0
production [0,2] = 0.0
production [0,3] = 300.0
production [0,4] = 200.0
production [1,0] = 200.0
production [1,1] = 100.0
production [1,2] = 0.0
production [1,3] = 0.0
production [1,4] = 100.0
production [2,0] = 0.0
production [2,1] = 0.0
production [2,2] = 400.0
production [2,3] = 300.0
production [2,4] = 0.0
Optimal objective value: 6500.0
"""

"""
Q3
Optimal solution:
production [0,0] = 0.0
production [0,1] = 0.0
production [0,2] = 0.0
production [0,3] = 0.0
production [0,4] = 0.0
production [1,0] = 200.0
production [1,1] = 100.0
production [1,2] = 0.0
production [1,3] = 300.0
production [1,4] = 300.0
production [2,0] = 0.0
production [2,1] = 0.0
production [2,2] = 400.0
production [2,3] = 300.0
production [2,4] = 0.0
fixed cost [0] = -0.0
fixed cost [1] = 1.0
fixed cost [2] = 1.0
Optimal objective value: 48800.0

Q4
Optimal solution:
production [0,0] = 0.0
production [0,1] = 0.0
production [0,2] = 0.0
production [0,3] = 0.0
production [0,4] = 300.0
production [1,0] = 200.0
production [1,1] = 100.0
production [1,2] = 400.0
production [1,3] = 0.0
production [1,4] = 0.0
production [2,0] = 0.0
production [2,1] = 0.0
production [2,2] = 0.0
production [2,3] = 600.0
production [2,4] = 0.0
fixed cost [0] = 1.0
fixed cost [1] = 1.0
fixed cost [2] = 1.0
  [0,0] = 0.0
  [0,1] = 0.0
  [0,2] = 0.0
  [0,3] = 0.0
  [0,4] = 1.0
  [1,0] = 1.0
  [1,1] = 1.0
  [1,2] = 1.0
  [1,3] = 0.0
  [1,4] = -0.0
  [2,0] = -0.0
  [2,1] = -0.0
  [2,2] = -0.0
  [2,3] = 1.0
  [2,4] = -0.0
Optimal objective value: 7800.0

"""







