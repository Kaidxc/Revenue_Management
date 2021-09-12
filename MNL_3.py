# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 13:13:08 2021

@author: ks4n19
"""

import numpy as np
import pandas as pd


q = {'q1':[30,35,40,45,50,40,40,40,40,40],'q2':[40,40,40,40,40,30,35,40,45,50]}
quantity = pd.DataFrame(q)

demand_rate = np.arange(105,160,5)

global theta, price, T, value

theta = np.random.uniform(0,1)
price = 1
value = np.array([2, 1.5])
v1 = value[0]
v2 = value[1]
T = 1

def uniform_cf(x):
    if x<0:
        Fx = 0
    elif 0<= x <=1:
        Fx = x
    elif x>1:
        Fx = 1
    
    return Fx

beta_1 = 1- uniform_cf(price/v1)


def revenue_separate(x,labda,q1,q2):
    if x>=0 and x<=v2/v1:
        alpha1_x = 1-uniform_cf(price*(1-x)/(v1-v2))
        alpha2_x = uniform_cf(price*(1-x)/(v1-v2))-uniform_cf(price*x/v2)
        t1_x = q1/(labda*alpha1_x)
        t2_x = q2/(labda*alpha2_x)
        
        if t1_x<=T and t2_x<=T:
            revenue_total_sparate = price*q1 + x*price*q2
        
        elif t1_x<=T and t2_x>T:
            beta2_x = 1-uniform_cf(x*price/v2)
            d2 = labda*alpha2_x*t1_x + labda*beta2_x*(T-t1_x)
            revenue_total_sparate = price*q1 + x*price*min(q2, d2)
        
        elif t1_x>T and t2_x<T:
            
            d1 = labda*alpha1_x*t2_x + labda*beta_1*(T-t2_x)
            revenue_total_sparate = x*price*q2 + price*min(q1, d1)
        
        elif t1_x>T and t2_x>T:
            revenue_total_sparate = price*labda*alpha1_x*T + x*price*labda*alpha2_x*T
    
    elif x> v2/v1 and x<=1:
        if q1>= labda*beta_1*T:
           revenue_total_sparate = price*labda*beta_1
         
        elif q1< labda*beta_1*T:
            d1 = q1
            d2 = labda*(1-uniform_cf(x*price/v2))*(T - q1/(labda*beta_1))
            revenue_total_sparate = price*d1 + x*price*d2
    return revenue_total_sparate


def revenue_fixed(x,labda,q1,q2):
    d = labda*T*(1-uniform_cf(x*price/((q1*v1 + q2*v2)/(q1 + q2))))
    revenue_total_fixed = x*price*(min(q1+q2, d))
    return revenue_total_fixed

    
def optimal_discount(x,labda,q1,q2):
    r_sparate = np.zeros(len(x))
    r_fixed = np.zeros(len(x))
    for i in range(len(x)):
        r_sparate[i] = revenue_separate(x[i],labda,q1,q2)
        r_fixed[i] = revenue_fixed(x[i],labda,q1,q2)
        
    r_sparate_list = r_sparate.tolist()
    r_fixed_list = r_fixed.tolist()
    opt_s = r_sparate_list.index(max(r_sparate_list))
    opt_f = r_fixed_list.index(max(r_fixed_list))
    
    return x[opt_s],r_sparate_list[opt_s] ,x[opt_f],r_fixed_list[opt_f] 
    
#====== under different lambda(demand_rate), with fixed quantity(40,40)=======
x = np.linspace(0,1,10000)
w = np.zeros(shape = (11,4))
for j in range(len(demand_rate)):
    w[j] = optimal_discount(x,demand_rate[j],40,40)

improve_1 = ((w[:,1]-w[:,3])/w[:,3])*100

df_1 = {'Lambda':demand_rate, 'optimal discount x': w[:,0], 'total revenue 1': w[:,1],
        'optimal discount y':w[:,2], 'total revenue 2':w[:,3],'imporvement(%)': improve_1}
output_1 = pd.DataFrame(df_1)

#======under different quantity, with fixed lambda(demand_rate=130)==========
e = np.zeros(shape = (len(quantity),4))
for t in range(len(quantity)):
    e[t] = optimal_discount(x,130,quantity.loc[t,'q1'],quantity.loc[t,'q2'])

improve_2 = 100*((e[:,1]-e[:,3])/e[:, 3])

    
df_2 = {'q_1': quantity['q1'], 'q_2': quantity['q2'], 'optimal discount x': e[:,0],
        'total revenue 1': e[:,1], 'optimal discount y': e[:,2], 'total revenue 2': e[:,3],
        'improvement(%)': improve_2}

output_2 = pd.DataFrame(df_2)


#output_1.to_excel('Pring_output.xlsx', sheet_name='sheet1', index=False)
#output_2 = output_1.copy()
#with pd.ExcelWriter('Pring_output.xlsx') as writer:  
    #output_1.to_excel(writer, sheet_name='sheet1')
    #output_2.to_excel(writer, sheet_name='sheet2')

print(output_1.to_latex(index=False))
print(output_2.to_latex(index=False))

