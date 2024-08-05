#!/usr/bin/env python
# coding: utf-8

# # Project: Men's Mile Run World Record Progression history analysis.

# In[3]:


# Required Libraries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import datetime as dt
pd.set_option('display.max_columns',None)   # None means no limit to columns to be displayed.


# In[4]:


# Reading the DataFrame:
df = pd.read_csv("Mens Mile Run Record Progression.csv")


#  Data Cleaning and Exploratory Data Analysis(EDA):

# In[5]:


df.head(10)


# In[6]:


df.shape


# In[7]:


df.describe(include='all')


# In[8]:


df.info()


# In[9]:


df.isna().sum()


# In[10]:


df.isna().any().sum()


# In[11]:


# Checking for duplicates:
df.drop_duplicates().shape          # No duplicate present


# In[12]:


# Filling out Null Values with a zero:
df=df.fillna(0)


# In[13]:


df.isna().any().sum()


# In[14]:


# Formatting time object to the regular string format:
df['Time']= df['Time'].str.replace('*', ' ', regex=False).str.replace('.', ':', regex=False)
  # regex=False so that expression is treated as a string (no interpretation of special xters)


# In[15]:


df['Time'].head()


# In[16]:


# Adding leading zero to single digit numbers (so that required format be adopted):


# In[17]:


df['Time']= df['Time'].str.replace(r'(\b\d\b)', r'0\1', regex=True)
  # regex=True so that pattern is treated as a regular expression involving interpretation of special xters.


# In[18]:


df['Time'].head()


# In[19]:


# Converting to datetime obj and also removing leading and trailing spaces (in order to match format):
df['Time']= pd.to_datetime(df['Time'].str.strip(), format= '%H:%M:%S')


# In[20]:


df['Time'].head()


# In[21]:


# Extracting hours, minutes and seconds (same time creating their columns in df):


# In[22]:


df['Hour']= df['Time'].dt.hour
df['Minute']=df['Time'].dt.minute
df['Second']= df['Time'].dt.second


# In[23]:


# Converting Time column to seconds:
      
df['Tot_Time_Seconds'] = df['Hour']*3600 + df['Minute'] * 60 + df['Second']                     


# In[24]:


df.head()


# In[25]:


# Average, minimum and maximum times for the mile run:
averg_time=df['Tot_Time_Seconds'].mean()
min_time = df['Tot_Time_Seconds'].min()
max_time = df['Tot_Time_Seconds'].max()


# In[26]:


print('Average mile run time is:',averg_time,'sec'),print('Minimum mile run time is:',min_time,'sec'), print('Maximum mile run time is:'\
                                                                                                 ,max_time,'sec')


# In[27]:


# Converting date column to datetime object:
from datetime import datetime
df['Date'] = df['Date'].str[:-3]  # To remove [5] at end of each datestring (string slicing).

df['Date']= pd.to_datetime(df['Date'], format= '%d %B %Y')


# In[28]:


df['Date'].head()


# In[29]:


df.head()


# In[30]:


# Remove abbreviations in venue to harmonize presentation:
df['Venue'] = df['Venue'].apply(lambda x: x.split(',')[0].strip() if isinstance(x,str) else x)


# In[31]:


df.head()


# In[32]:


# Drop the Auto column and maintain df:
df.drop(columns=['Auto'], inplace=True)


# In[33]:


df.head()


# Dataset Analysis:

# In[34]:


df.drop(columns=['index','Hour', 'Minute', 'Second'], inplace=True)


# In[35]:


df.head()


# In[36]:


print(df.dtypes)


# In[37]:


df.isna().any().sum()


# In[38]:


df.dropna(axis=0, inplace=True)    # axis=0 means drop row with null value, in place, thus no new df created.


# In[39]:


df.head()


# In[40]:


# Distribution of total mile run time:
sns.histplot(df['Tot_Time_Seconds'],bins=10, edgecolor='red')


# In[41]:


# A bar plot of athlete vs total run time:

sns.barplot(x='Athlete', y='Tot_Time_Seconds', data=df)
plt.title('Total time run in secs vs Athlete')
plt.xticks(rotation=90)
plt.show()


# In[160]:


# Grouping athletes according to nationality:


# In[41]:


athl_nat = df.groupby('Athlete')[['Nationality']].first().reset_index()


# In[43]:


athl_nat                    # first() means grouping output in two columns with athlete first
                           # reset_index() used to turn the grouped indices to columns.


# In[42]:


# Grouping athletes with more than 1 NationLity:
athl_Natmore1= df.groupby('Athlete')['Nationality'].agg(list).reset_index()


# In[45]:


athl_Natmore1


# In[43]:


# Grouping Athletes by United Kingdom nationality:
athlt_Uk = athl_nat.groupby('Nationality').filter(lambda x: 'United Kingdom' in x['Nationality'].values)


# In[47]:


athlt_Uk


# In[44]:


# Extracting year component:
df['Year']= df['Date'].dt.year


# In[45]:


df['Year'].head()


# In[46]:


# A line plot to show the variation of run time over the years:
plt.figure(figsize=(10,6))
sns.set_style("whitegrid")
sns.lineplot(x='Year', y= 'Tot_Time_Seconds', data=df, marker='o', color='b')
plt.xlabel("Years")
plt.ylabel("Total Time in Secs")
plt.title("Mens Mile run time vs years")
plt.show()


# In[47]:


# The line plot shows a decline in run time as over the years translating improved performance of athletes.


# In[47]:


# Athletes with the best performance:
athl_bestPrf= df.groupby("Athlete")[["Tot_Time_Seconds"]].max().sort_values(by='Tot_Time_Seconds')


# In[52]:


athl_bestPrf


# In[ ]:





# In[51]:


# Determine the best performed and least performed Athlete:


# In[48]:


overall_bestAthl = athl_bestPrf.idxmin()


# In[49]:


print("Overall best performed athlete is:",overall_bestAthl)


# In[50]:


overall_leastAthl = athl_bestPrf.idxmax()


# In[51]:


print("Overall least performed athlete is:",overall_leastAthl)


# In[52]:


sns.barplot( x='Nationality', y= 'Tot_Time_Seconds',data=df)
plt.title('Total Run Time in Seconds vs Nationality')
plt.xticks(rotation=90)
plt.show()


# In[53]:


df1 = df.rename(columns={'Nationality': 'Country', 'Tot_Time_Seconds': 'Performance'})
df1.head()


# In[54]:


# Grouping countries by total number of seconds run (Performance):
countries_grpd= df1.groupby('Performance')[['Country']].first().sort_values(by='Performance')
countries_grpd


# In[55]:


# From the grouping above, the best athlete came from Morocco.


# In[56]:


# Grouping performance by venue:
grouped_prf = df1.groupby('Venue')[['Performance']].first().sort_values(by='Performance')
grouped_prf


# In[62]:


# From the above grouping, the best performance was recorded in Rome.


# In[57]:


# Determine venue with hieghest number of athletes:
grpd_athl_perVenue = df.groupby('Venue')[['Athlete']].nunique().sort_values(by='Athlete')
grpd_athl_perVenue


# In[58]:


# Find country with maximum number of athletes

# Number of Athletes per country:
athl_perCntry= athl_nat.groupby("Nationality")["Athlete"].count().sort_values(ascending=False)
athl_perCntry


# In[59]:


print("Country with maximum number of athletes is: United Kingdom, having 6 athlets.")


# In[60]:


print("Country with the least number of athletes is: Tanzania, having just 1 athlete.")


# In[61]:


# Grouping athlete per venue, per year with corresponding total time in secs(Performance):
grouped_prf2 = df1.groupby(['Venue', 'Year', 'Performance']).agg({'Athlete' : 'nunique'}).reset_index()
grouped_prf2


# In[62]:


# Using the above sub-dataFrame(grouped_prf2) to plot a graph of performance vs venue:

sns.barplot(x= 'Venue', y= 'Performance', data= grouped_prf2)
plt.title('Performance over years based on venue')
plt.xticks(rotation=90)

plt.show()


# In[1]:


# The plot above confirms that the best performance was in Rome ( lowest Tot_time_Seconds of run).


# In[70]:


# Checking for any correlation in dataset:


# In[67]:


sns.pairplot(data=df1)   # That is the index column is excluded


# In[88]:


# There is an observed negative correlation between year and total time run in seconds(Performance) from the plot above.


# In[68]:


# Checking any correlation between venue and total time of run (Performance):

sns.scatterplot(x= "Venue", y="Performance", data=grouped_prf2)
plt.title('Scattered plot of Total Time Run vs Venue')
plt.xticks(rotation=90)
plt.show()


# In[84]:


# No correlation observed in the plot above.


# In[69]:


# Box plot to check for outliers:
sns.boxplot(x='Year', y='Performance', data=df1)
plt.title('Box plot of Total Run Time in Seconds vs Years')
plt.xticks(rotation=45)
plt.show()


# In[104]:


# No outliers identified.


# #**Predictive Analyss** (Linear Regression):

# In[63]:


# Determine future world record run times based on available data:
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split


# In[64]:


# Separating variables:
X = df1[['Year']].values
y = df1['Performance'].values


# In[65]:


# Splitting the data into training and test sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[66]:


# Instantiate and fit model:
lrm = LinearRegression()
lrm.fit(X_train,y_train)


# In[67]:


# Predict on the test set:
y_pred = lrm.predict(X_test)


# In[68]:


# Evaluate the model on the test data:

# Calculate performance metrics:
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


# In[69]:


print(f"Mean Absolute Error (MAE): {mae:.3f}")
print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
print(f"R^2 Score: {r2:.3f}")


# In[82]:


# The values of the metrics above show that the model is performing very well in predicting the performance of athletes in
# future runs. An R-Squared score of 0.964 (96.4%) means that 96.4% of variations in the dependent variable can be eplained
# by the independent variable ('Year'). Hence, suggesting that the model fits the data very well and predictions are very
# close to actual values.The model captured most of the variability in the data.


# In[70]:


# Predictions based on available data:

future_yrs = np.array([[2010], [2015], [2020]])

predictions = lrm.predict(future_yrs)
predictions


# In[72]:


# Plot of actual vs Predicted values:

plt.figure(figsize=(8, 5))
plt.plot(df['Year'], y, label='Actual', marker='o')
plt.scatter(future_yrs, predictions, color='red', label='Predicted (Future)', zorder=5)
plt.xlabel('Year')
plt.ylabel('Total Time in Seconds')
plt.title('Men\'s Mile Run World Record Progression: Actual vs Predicted')
plt.legend()
plt.show()


# In[2]:


# The predictions show a clear decline in the total time of the runs over the years, indicating improved performance 
# by athletes.This may probably be due to improved training infrastructure, improved diets and more motivation 
#to the athletes.


# In[78]:


# These predictions can help other younger athletes to set achievable goals in their early careers.


# In[ ]:




