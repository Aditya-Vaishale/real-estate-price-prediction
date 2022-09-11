#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing necessary library
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
matplotlib.rcParams["figure.figsize"]=(10,20)


# In[2]:


#locating dataset and importing it in code
os.chdir('D:\\real estate price prediction')
df1= pd.read_csv("Bengaluru_House_Data.csv")
df1.head()


# In[3]:


#dropping the unnecessary columns
df2= df1.drop(['area_type','society','balcony','availability'], axis='columns')
df2.head()


# In[4]:


#checking the null values 
df2.isnull().sum()


# In[5]:


#removing the null values 
df3=df2.dropna()
df3.isnull().sum()


# In[6]:


#adding the bhk column
df3['bhk']=df3['size'].apply(lambda x: int(x.split(' ')[0]))


# In[7]:


df3.head()


# In[8]:


df3['bhk'].unique()


# In[9]:


df3[df3.bhk>20]


# In[10]:


df3.total_sqft.unique()


# In[11]:


#creating a function to get "1133-1384" values...
def is_float(z):
    try:
        float(z)
    except:
        return False
    return True


# In[12]:


df3[~df3.total_sqft.apply(is_float)].head(10)


# In[13]:


# after listing such values creating a function so that we can treat them  
def convert_sqft_to_num(x):
	token = x.split('-')
	if len(token) == 2:
		return (float(token[0])+float(token[1]))/2
	try:
		return float(x)
	except:
		return None
                


# In[14]:


#testing convert_sqft_to_num()function
df4 = df3.copy()
df4.total_sqft = df4.total_sqft.apply(convert_sqft_to_num)
df4.head(3)


# In[15]:


#we are creating price_per_sqft to remove some outliers
df5= df4.copy()
df5['price_per_sqft']=df5['price']*100000/df5['total_sqft']
df5.head()


# In[16]:


len(df5.location.unique())


# In[17]:


# we are using creating location_stat so that we can get stats about how many data point are ther in that location
df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending = False)
location_stats


# In[18]:


len(location_stats[location_stats<=10])


# In[19]:


#separating the location which are having less than 10 data points
location_stats_less_than_10= location_stats[location_stats<=10]
location_stats_less_than_10


# In[20]:


len(df5.location.unique())


# In[21]:


#creating other column which will for location_stats_less_than_10
df5.location= df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
len(df5.location.unique())


# In[22]:


df5.head()


# In[23]:


#these are outlier that we need to safely remove.
df5[df5.total_sqft/df5.bhk<300].head()


# In[24]:


#so we removed that outlier
df6 = df5[~(df5.total_sqft/df5.bhk<300)]
df6.shape


# In[25]:


df6.price_per_sqft.describe()


# In[26]:


#this function will remove the max() of price_per_sqft
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st))&(subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index= True)
    return df_out

df7 = remove_pps_outliers(df6)
df7.shape


# In[27]:


def plot_scatter_chart(df, location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,marker='*',color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+',color='red',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price")
    plt.title(location)
    plt.legend()
    
plot_scatter_chart(df7, "Hebbal")


# In[28]:


#this function will remove outlier which are having mean less than its previous bhk 
def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk]={
            'mean' : np.mean(bhk_df.price_per_sqft), 'std': np.std(bhk_df.price_per_sqft), 'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])]. index.values)
    return df.drop(exclude_indices, axis='index')   

df8 = remove_bhk_outliers(df7)
df8.shape


# In[29]:


plot_scatter_chart(df8, "Hebbal")


# In[30]:


matplotlib.rcParams["figure.figsize"]=(20,10)
plt.hist(df8.price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")


# In[31]:


df8.bath.unique()


# In[32]:


df8[df8.bath>10]


# In[33]:


plt.hist(df8.bath, rwidth=0.8)
plt.xlabel("Number of Bathroom")
plt.ylabel("Count")


# In[34]:


df8[df8.bath>df8.bhk+2]


# In[35]:


df9 = df8[df8.bath<df8.bhk+2]
df9.shape


# In[36]:


df10 = df9.drop(['size', 'price_per_sqft'], axis='columns')
df10.head()


# In[37]:


#after this we will implement algorithm, because now we have a desired data


# In[38]:


# we cant give a text directly as a model we need to convert it into integers 
dummies =pd.get_dummies(df10.location)
dummies.head(10)


# In[39]:


#we are dropping the other column to prevent from dummies trap
df11 = pd.concat([df10, dummies.drop('other', axis='columns')], axis='columns')
df11.head()


# In[40]:


df12 = df11.drop('location', axis='columns')
df12.head()


# In[41]:


df12.shape


# In[42]:


#this contains all independent variable
X = df12.drop('price', axis='columns')
X.head()


# In[43]:


#this contains dependent variables
y=df12.price
y.head()


# In[44]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=10)


# In[45]:


from sklearn.linear_model import LinearRegression
lr_clf= LinearRegression()
lr_clf.fit(X_train, y_train)
lr_clf.score(X_test, y_test)


# In[46]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
cross_val_score(LinearRegression(), X,y, cv=cv)


# In[47]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model(X, y):
    algo = {
        'linear_regression':{
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        
        'lasso':{
            'model': Lasso(),
            'params':{
                'alpha':[1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree':{
            'model': DecisionTreeRegressor(),
            'params':{
                'criterion': ['mse', 'friedman_mse'],
                'splitter': ['best', 'random']
            }
        }
    }
    
    scores= []
    cv= ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algo.items():
        gs= GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model':algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })
    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])    

find_best_model(X,y)


# In[48]:


def predict_price(location,sqft,bath,bhk):
    loc_index = np.where(X.columns==location)[0][0]
    
    x= np.zeros(len(X.columns))
    x[0]= sqft
    x[1]= bath
    x[2]= bhk
    if loc_index >=0:
        x[loc_index] = 1
        
    return lr_clf.predict([x*100000])[0]


# In[49]:


predict_price('1st Phase JP Nagar', 1000, 2, 2)


# In[50]:


predict_price('Kothanur', 1000, 2, 3)


# In[51]:


predict_price('Hebbal', 510, 2, 2)


# In[52]:


predict_price('Whitefield', 1000, 2, 2)


# In[53]:


predict_price('Whitefield', 2000, 4, 3)


# In[ ]:




