#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import math
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# # Reading IPL 2020 dataset

# In[2]:


data_2020 = pd.read_csv('ipl_20.csv')
data_2020.head()


# # Checking for null values in IPL 2020 dataset

# In[3]:


missing_value=["Null"]
df=pd.read_csv('ipl_20.csv' , na_values=missing_value)
df.isnull().sum()


# # Dropping all rows having null values and saving into a new csv file IPL_2020.csv

# In[4]:


df = df.dropna()


# In[5]:


df.to_csv('IPL_2020.csv')


# In[6]:


missing_value=["Null"]
df=pd.read_csv('IPL_2020.csv' , na_values=missing_value)
df.isnull().sum()


# In[7]:


data_2020 = pd.read_csv('IPL_2020.csv')
data_2020


# In[8]:


data_2020 = pd.read_csv('IPL_2020.csv')
data_2020['CBR']=data_2020['CBR'].str.replace('-','0')
data_2020['log10(CBR)']=data_2020['log10(CBR)'].str.replace('-','2')
data_2020['Bowling Strike Rate']=data_2020['Bowling Strike Rate'].str.replace('-','0')
data_2020['Bowling Average']=data_2020['Bowling Average'].str.replace('-','0')
data_2020


# In[9]:


data_2020['log10(CBR)']=data_2020['log10(CBR)'].astype(float)
data_2020['CBR']=data_2020['CBR'].astype(float)
data_2020['Bowling Strike Rate']=data_2020['Bowling Strike Rate'].astype(float)
data_2020['Bowling Average']=data_2020['Bowling Average'].astype(float)


# In[10]:


data_2020.info()


# # Reading IPL 2021 dataset

# In[11]:


data_2021 = pd.read_csv('ipl_21.csv')
data_2021.head()


# # Checking for null values in IPL 2021 dataset

# In[12]:


missing_value=["Null"]
df=pd.read_csv('ipl_21.csv' , na_values=missing_value)
df.isnull().sum()


# # Dropping all rows having null values and saving into a new csv file IPL_2021.csv

# In[13]:


df = df.dropna()


# In[14]:


df.to_csv('IPL_2021.csv')


# In[15]:


missing_value=["Null"]
df=pd.read_csv('IPL_2021.csv' , na_values=missing_value)
df.isnull().sum()


# In[16]:


data_2021 = pd.read_csv('IPL_2021.csv')
data_2021['CBR']=data_2021['CBR'].str.replace('-','0')
data_2021['log10(CBR)']=data_2021['log10(CBR)'].str.replace('-','2')
data_2021['Bowling Average']=data_2021['Bowling Average'].str.replace('-','0')
data_2021['Bowling Strike Rate']=data_2021['Bowling Strike Rate'].replace('-','0')
data_2021


# In[17]:


data_2021['log10(CBR)']=data_2021['log10(CBR)'].astype(float)
data_2021['CBR']=data_2021['CBR'].astype(float)
data_2021['Bowling Strike Rate']=data_2021['Bowling Strike Rate'].astype(float)
data_2021['Bowling Average']=data_2021['Bowling Average'].astype(float)


# In[18]:


data_2021.info()


# # Dropping the 'unnamed' and 'S.No.' column from the IPL data, since it doesn't contribute the target variable of predicting the IPL player value

# In[19]:


data_2020.drop(['Unnamed: 0','S.No.'],axis=1,inplace=True)
data_2020.info()


# In[20]:


data_2021.drop(['Unnamed: 0','S. No.'],axis=1,inplace=True)
data_2021.info()


# # Cleaned 2020 dataset

# In[21]:


data_2020['Salary'] = data_2020['Salary'].str.replace(',','').str.replace('$', '').astype(float)
data_2020['Value']=np.abs(data_2020['Value'])
data_2020['Salary']=np.abs(data_2020['Salary'])
data_2020['RAA']=np.abs(data_2020['RAA'])
data_2020['Wins']=np.abs(data_2020['Wins'])
data_2020=data_2020.groupby('Name').sum().reset_index()


# In[22]:


data_2020


# # Cleaned 2021 dataset

# In[23]:


data_2021['Salary'] = data_2021['Salary'].str.replace(',','').str.replace('$', '').astype(float)
data_2021['Value']=np.abs(data_2021['Value'])
data_2021['Salary']=np.abs(data_2021['Salary'])
data_2021=data_2021.groupby('Name').sum().reset_index()


# In[24]:


data_2021


# # Data Visualization

# # Concatinating both dataset into a new dataset for visualization purposes

# In[25]:


ipl_full=pd.concat([data_2020,data_2021],ignore_index=True)
ipl_full


# In[26]:


ipl_full.info()


# In[27]:


ipl_full["RAA"]=ipl_full["RAA"].astype(float)
ipl_full.info()


# In[28]:


ipl_full


# # Grouping full IPL dataset by name

# In[29]:


df_full=ipl_full.groupby('Name').sum().reset_index()
df_full.to_csv('dffull.csv')
df_full


# # Some values become insignificant since they are not addable (eg. Economy rate, Average, Strike rate, CBR etc.)
# # So, computing proper data

# In[30]:


df_full['Bowling Average']=df_full['Runs Conceded']/df_full['Wickets taken']
df_full['Bowling Strike Rate']=df_full['No. of balls']/df_full['Wickets taken']


# In[31]:


df_full['Economy rate']=df_full['Runs Conceded']/df_full['No. of Overs']
df_full['CBR']=3/((1/df_full['Bowling Average'])+(1/df_full['Bowling Strike Rate'])+(1/df_full['Economy rate']))


# In[32]:


df_full['log10(CBR)']=np.log10(df_full['CBR'])


# # Making salary and value as positive since negative salary or value is meaningless

# In[33]:


df_full['Value']=np.abs(df_full['Value'])
df_full['Salary']=np.abs(df_full['Salary'])


# # Bar plots for full dataset

# # Total Matches Played vs Bowlers

# In[34]:


bars_heights = df_full['Matches']
bars_label = df_full['Name']
plt.subplots(figsize=(20,10))
plt.xlabel('Name of Bowlers')
plt.ylabel('Matches')
plt.bar(range(len(bars_label)), bars_heights,color=['red', 'green','orange', 'blue', 'cyan','yellow'])
plt.xticks(range(len(bars_label)), bars_label, rotation='vertical')
plt.show()


# # Total Wickets Taken vs Bowlers

# In[35]:


bars_heights = df_full['Wickets taken']
bars_label = df_full['Name']
plt.subplots(figsize=(20,10))
plt.xlabel('Name of Bowlers')
plt.ylabel('Wickets taken')
plt.bar(range(len(bars_label)), bars_heights,color=['red', 'green','orange', 'blue', 'cyan','yellow'])
plt.xticks(range(len(bars_label)), bars_label, rotation='vertical')
plt.show()


# # Total Value vs Bowlers

# In[36]:


bars_heights = df_full['Value']
bars_label = df_full['Name']
plt.subplots(figsize=(20,10))
plt.xlabel('Name of Bowlers')
plt.ylabel('Value')
plt.bar(range(len(bars_label)), bars_heights,color=['red', 'violet','orange', 'blue', 'cyan','yellow'])
plt.xticks(range(len(bars_label)), bars_label, rotation='vertical')
plt.show()


# # log10(CBR) vs Bowlers

# In[37]:


bars_heights = df_full['log10(CBR)']
bars_label = df_full['Name']
plt.subplots(figsize=(20,10))
plt.xlabel('Name of Bowlers')
plt.ylabel('log10(CBR)')
plt.bar(range(len(bars_label)), bars_heights,color=['black','red', 'green','orange', 'blue', 'cyan','yellow'])
plt.xticks(range(len(bars_label)), bars_label, rotation='vertical')
plt.show()


# # Economy Rate vs Bowlers

# In[38]:


bars_heights = df_full['Economy rate']
bars_label = df_full['Name']
plt.subplots(figsize=(20,10))
plt.xlabel('Name of Bowlers')
plt.ylabel('Economy rate')
plt.bar(range(len(bars_label)), bars_heights,color=['black','red', 'green','orange', 'blue', 'cyan','yellow'])
plt.xticks(range(len(bars_label)), bars_label, rotation='vertical')
plt.show()


# # Bowling Strike Rate vs Bowlers

# In[39]:


bars_heights = df_full['Bowling Strike Rate']
bars_label = df_full['Name']
plt.subplots(figsize=(20,10))
plt.xlabel('Name of Bowlers')
plt.ylabel('Bowling Strike Rate')
plt.bar(range(len(bars_label)), bars_heights,color=['black','red', 'green','orange', 'blue', 'cyan','yellow'])
plt.xticks(range(len(bars_label)), bars_label, rotation='vertical')
plt.show()


# # Bowling Average vs Bowlers

# In[40]:


bars_heights = df_full['Bowling Average']
bars_label = df_full['Name']
plt.subplots(figsize=(20,10))
plt.xlabel('Name of Bowlers')
plt.ylabel('Bowling Average')
plt.bar(range(len(bars_label)), bars_heights,color=['black','red', 'green','orange', 'blue', 'cyan','yellow'])
plt.xticks(range(len(bars_label)), bars_label, rotation='vertical')
plt.show()


# # Analyzing log10(CBR) value and deciding good, average and poor performance of bowlers

# In[41]:


good = df_full.loc[df_full['log10(CBR)'] <1.125]
good


# In[42]:


poor = df_full.loc[df_full['log10(CBR)'] >1.193]
poor


# In[43]:


good_poor= pd.concat([good, poor],ignore_index=True)
good_poor


# In[44]:


average= pd.concat([df_full,good_poor]).drop_duplicates(keep=False)
average


# In[45]:


from pydataset import data
from scipy.stats import norm


# # Histogram for Economy rate

# In[46]:


get_ipython().run_line_magic('matplotlib', 'inline')
hist=df_full.hist('Economy rate')
plt.show()


# # Normal distribution curve for Economy rate

# In[47]:


get_ipython().run_line_magic('matplotlib', 'inline')
x=df_full['Economy rate']
sns.displot(x, kde=True)


# #  Normal distribution curve for log10(CBR)

# In[48]:


get_ipython().run_line_magic('matplotlib', 'inline')
x=df_full['log10(CBR)']
sns.displot(x, kde=True)


# # Keeping only CBR , RAA...etc. values in stats_last dataframe

# In[49]:


stats_last=df_full.drop(['Name','Matches','No. of balls','No. of Overs','Wickets taken','Runs Conceded','Economy rate','Dot Balls','Maiden overs','Bowling Strike Rate','Bowling Average'],axis=1)
stats_last


# # Plotting pairplots for stats_last

# In[50]:


sns_plot=sns.pairplot(data=stats_last)
sns_plot


# # Data statistics for different dataframes
# # For IPL 2020

# In[51]:


data_2020.describe()


# # For IPL 2021

# In[52]:


data_2021.describe()


# # For both seasons combined

# In[53]:


df_full.describe()


# # Hypothesis Testing

# In[54]:


df_full


# In[55]:


get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.figure_factory as ff
import plotly.graph_objects as go
fig = ff.create_distplot([data_2020['Economy rate'],data_2021['Economy rate']],['2020 Economy','2021 Economy'],curve_type='normal')
fig.update_layout(title='Distribution for economy rate in IPL 2020 and 2021',xaxis_title='Economy rate',yaxis_title='Density')
fig.show()


# In[56]:


import random
test_list = df_full['Economy rate']
list=[]
for i in range(0,10):
    rand_idx = random.randint(0, len(test_list)-1)
    random_num = test_list[rand_idx]
    list.append(random_num)
eco=pd.DataFrame(list)
print ("Sample : ")
eco


# In[57]:


sampMean=eco.mean()
sampMean


# In[58]:


df_full.describe()


# # Null Hypothesis : Average economy rate of sample is same as that of population
# # Alternative Hypothesis : Average economy rate of sample is less than that of population

# In[59]:


from scipy import stats
from statsmodels.stats import weightstats as stests
pop_mean=np.mean(df_full['Economy rate'])
print("Population mean is :- ", pop_mean)
print("Sample mean is :- ",sampMean)
ztest_Score, p_value= stests.ztest(eco,value = pop_mean, alternative='smaller')
# the function outputs a p_value and z-score corresponding to that value, we compare the 
# p-value with alpha, if it is greater than alpha then we do not reject null hypothesis 
# else we reject it.
alpha=0.05
print("alpha is :- ",alpha)
print("z score is :- ",ztest_Score)
print("p value is :- ",p_value)
if(p_value <  alpha):
  print("Reject Null Hypothesis")
else:
  print("Failed to Reject Null Hypothesis")


# In[60]:


import random
test_list = df_full['Value']
list=[]
for i in range(0,10):
    rand_idx = random.randint(0, len(test_list)-1)
    random_num = test_list[rand_idx]
    list.append(random_num)
value=pd.DataFrame(list)
print ("Sample : ")
value


# In[61]:


sampMean=value.mean()
sampMean


# # Null Hypothesis : Average value of sample is same as that of population
# # Alternative Hypothesis : Average value of sample is less than that of population

# In[62]:


from scipy import stats
from statsmodels.stats import weightstats as stests
pop_mean=np.mean(df_full['Value'])
print("Population mean is :- ", pop_mean)
print("Sample mean is :- ",sampMean)
ztest_Score, p_value= stests.ztest(value,value = pop_mean, alternative='smaller')
# the function outputs a p_value and z-score corresponding to that value, we compare the 
# p-value with alpha, if it is greater than alpha then we do not reject null hypothesis 
# else we reject it.
print("alpha is :- ",alpha)
print("z score is :- ",ztest_Score)
print("p value is :- ",p_value)
if(p_value <  alpha):
  print("Reject Null Hypothesis")
else:
  print("Failed to Reject Null Hypothesis")


# In[63]:


get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.figure_factory as ff
import plotly.graph_objects as go
fig=ff.create_distplot([data_2020['Value'],data_2021['Value']],['2020 Value','2021 Value'],curve_type='normal',show_hist=True)
fig.update_layout(title='Distribution for value in IPL 2020 and 2021',xaxis_title='Value',yaxis_title='Density',yaxis_range=[0,0.00000002])
fig.show()


# # Prediction Model through Machine Learning (ML)

# In[64]:


#Importing the required modules for ML model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# # Prediction of IPL 2020 bidding price based on bowlers performance and comparing them with actual values

# In[65]:


data_2020


# In[66]:


train = data_2020.drop(['Name', 'Value', 'No. of balls','No. of Overs'],axis=1)
test= data_2020['Value']


# In[67]:


# Using test_train_split method to split the dataset into testing and training dataset
# Sample size is 0.3 of total i.e 0.3*80= 24 players
X_train, X_test, y_train, y_test = train_test_split(train, test, test_size=0.3, random_state=1)


# In[68]:


regr=LinearRegression()


# In[69]:


regr.fit(X_train, y_train)


# In[70]:


pred=regr.predict(X_test)


# In[71]:


pred


# In[72]:


# Printing accuracy of our prediction model
regr.score(X_test, y_test)


# # 78.43 % accuracy

# In[73]:


#Comparing actual prices with predicted prices
compare = pd.DataFrame({'Actual': y_test, 'Predicted': np.abs(pred)})
compare


# # Regression plot for IPL 2020 bidding prices

# In[74]:


x_train = X_train.values[:,0].reshape(-1, 1)
x_test = X_test.values[:,0].reshape(-1, 1)
simple_reg = LinearRegression()
simple_reg.fit(x_train, y_train)
y_pred = simple_reg.predict(x_test)
plt.scatter(x_test, y_test,  color='black')
plt.plot(x_test, y_pred, color='blue', linewidth=3)


# In[75]:


regr_model= regr.fit(X_train, y_train)
regr_predictions= regr.predict(X_test)
plt.scatter(y_test, regr_predictions)
plt.title("Plot of Predicted value vs Actual Value")
plt.xlabel("Actual Value")
plt.ylabel("Predicted value")
z = np.polyfit(y_test, regr_predictions, 1)
p = np.poly1d(z)
plt.plot(y_test,p(y_test),"r--")
plt.show()
plt.close()


# In[76]:


#Pairwise correlation between different features
corr = data_2020.corr()
corr.style.background_gradient(cmap='coolwarm')


# # Prediction of IPL 2021 bidding price based on bowlers performance and comparing them with actual values

# In[77]:


data_2021


# In[78]:


train = data_2021.drop(['Name', 'Value', 'No. of balls','No. of Overs'],axis=1)
test= data_2021['Value']


# In[79]:


# Using test_train_split method to split the dataset into testing and training dataset
# Sample size is 0.3 of total i.e 0.3*80= 24 players
X_train, X_test, y_train, y_test = train_test_split(train, test, test_size=0.3, random_state=2)


# In[80]:


regr=LinearRegression()


# In[81]:


regr.fit(X_train, y_train)


# In[82]:


pred=regr.predict(X_test)


# In[83]:


pred


# # 94.4% accuracy

# In[84]:


# Printing accuracy of our prediction model
regr.score(X_test, y_test)


# In[85]:


#Comparing actual prices with predicted prices
compare = pd.DataFrame({'Actual': y_test, 'Predicted': np.abs(pred)})
compare


# # Regression plot for IPL 2021 bidding prices

# In[86]:


x_train = X_train.values[:,0].reshape(-1, 1)
x_test = X_test.values[:,0].reshape(-1, 1)
simple_reg = LinearRegression()
simple_reg.fit(x_train, y_train)
y_pred = simple_reg.predict(x_test)
plt.scatter(x_test, y_test,  color='black')
plt.plot(x_test, y_pred, color='blue', linewidth=3)


# In[87]:


regr_model= regr.fit(X_train, y_train)
regr_predictions= regr.predict(X_test)
plt.scatter(y_test, regr_predictions)
plt.title("Plot of Predicted value vs Actual Value")
plt.xlabel("Actual Value")
plt.ylabel("Predicted value")
z = np.polyfit(y_test, regr_predictions, 1)
p = np.poly1d(z)
plt.plot(y_test,p(y_test),"r--")
plt.show()
plt.close()


# In[88]:


#Pairwise correlation between different features
corr = data_2021.corr()
corr.style.background_gradient(cmap='coolwarm')


# # Prediction of average bidding price based on bowlers performance in IPL 2020 and 2021, and comparing them with actual values

# In[89]:


df_full= pd.read_csv('df_full_average.csv')
df_full= df_full.drop(['Unnamed: 0'], axis=1)
df_full


# In[90]:


train = df_full.drop(['Name', 'Value', 'No. of balls','No. of Overs'],axis=1)
test= df_full['Value']


# In[91]:


# Using test_train_split method to split the dataset into testing and training dataset
# Sample size is 0.3 of total i.e 0.3*80= 24 players
X_train, X_test, y_train, y_test = train_test_split(train, test, test_size=0.3, random_state=1)


# In[92]:


regr=LinearRegression()


# In[93]:


regr.fit(X_train, y_train)


# In[94]:


pred=regr.predict(X_test)


# In[95]:


pred


# # 62.17 % accuracy

# In[96]:


# Printing accuracy of our prediction model
regr.score(X_test, y_test)


# In[97]:


#Comparing actual prices with predicted prices
compare = pd.DataFrame({'Actual': y_test, 'Predicted': np.abs(pred)})
compare


# # Regression plot for average bidding prices

# In[98]:


x_train = X_train.values[:,0].reshape(-1, 1)
x_test = X_test.values[:,0].reshape(-1, 1)
simple_reg = LinearRegression()
simple_reg.fit(x_train, y_train)
y_pred = simple_reg.predict(x_test)
plt.scatter(x_test, y_test,  color='black')
plt.plot(x_test, y_pred, color='blue', linewidth=3)


# In[99]:


regr_model= regr.fit(X_train, y_train)
regr_predictions= regr.predict(X_test)
plt.scatter(y_test, regr_predictions)
plt.title("Plot of Predicted value vs Actual Value")
plt.xlabel("Actual Value")
plt.ylabel("Predicted value")
z = np.polyfit(y_test, regr_predictions, 1)
p = np.poly1d(z)
plt.plot(y_test,p(y_test),"r--")
plt.show()
plt.close()


# In[100]:


#Pairwise correlation between different features
corr = df_full.corr()
corr.style.background_gradient(cmap='coolwarm')


# In[101]:


#Importing the required modules for ML model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Prediction of average bidding price based on bowlers performance in IPL 2020 and 2021, and comparing them with actual values
train = df_full.drop(['Name', 'Value', 'No. of balls','No. of Overs'],axis=1)
test= df_full['Value']

# Using test_train_split method to split the dataset into testing and training dataset
# Sample size is 0.3 of total i.e 0.3*80= 24 players
X_train, X_test, y_train, y_test = train_test_split(train, test, test_size=0.3, random_state=1)

regr=LinearRegression()
regr.fit(X_train, y_train)
pred=regr.predict(X_test)
pred

# Printing accuracy of our prediction model
regr.score(X_test, y_test)

#Comparing actual prices with predicted prices
compare = pd.DataFrame({'Actual': y_test, 'Predicted': np.abs(pred)})
compare

# Regression plot for average bidding prices for both IPL 2020 and 2021 seasons
x_train = X_train.values[:,0].reshape(-1, 1)
x_test = X_test.values[:,0].reshape(-1, 1)
simple_reg = LinearRegression()
simple_reg.fit(x_train, y_train)
y_pred = simple_reg.predict(x_test)
plt.scatter(x_test, y_test,  color='black')
plt.plot(x_test, y_pred, color='blue', linewidth=3)

#Plot of Predicted value vs Actual Value
regr_model= regr.fit(X_train, y_train)
regr_predictions= regr.predict(X_test)
plt.scatter(y_test, regr_predictions)
plt.title("Plot of Predicted value vs Actual Value")
plt.xlabel("Actual Value")
plt.ylabel("Predicted value")
z = np.polyfit(y_test, regr_predictions, 1)
p = np.poly1d(z)
plt.plot(y_test,p(y_test),"r--")
plt.show()
plt.close()

