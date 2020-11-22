---
title: "Happiness 2017 using Linear Regression"
date: 2020-09-13
tags: [Applied Data Science]
header:
  image: "/images/happiness.jpg"
excerpt: "Data Science, Data Wrangling, Messy Data"
mathjax: "true"
---

# Project 1:
## DSC680 Applied Data Science						


### project 1- DSC680 

### Happiness 2017
soukhna Wade
09/18/2020

### Introduction

There are three parts to my report as follows:

** Cleaning
** Visualization
** Prediction

The purpose of choosing this work is to find out which factors are more important to live a happier life. As a result, people and countries can focus on the more significant factors to achieve a higher happiness level. We also will implement several machine learning algorithms to predict the happiness score and compare the result to discover which algorithm works better for this specific dataset.

https://www.kaggle.com/pinarkaya/world-happiness-eda-visualization-ml/data#Linear-Regression
              
https://www.kaggle.com/sarahvch/investigating-happiness-with-python/execution#Setting-up-Linear-Model-to-Predict-Happiness
              
              

### Import necessary Libraries


```python
# Standard library import-Python program# for some basic operations
import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt    # for graphics
import seaborn as sns              # for visualizations
plt.style.use('fivethirtyeight')                

import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Use to configure display of graph
%matplotlib inline 

#stop unnecessary warnings from printing to the screen
import warnings
warnings.simplefilter('ignore')

# for interactive visualizations
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode(connected = True)
```


<script type="text/javascript">
window.PlotlyConfig = {MathJaxConfig: 'local'};
if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
if (typeof require !== 'undefined') {
require.undef("plotly");
requirejs.config({
    paths: {
        'plotly': ['https://cdn.plot.ly/plotly-latest.min']
    }
});
require(['plotly'], function(Plotly) {
    window._Plotly = Plotly;
});
}
</script>



### Import and read Dataset from local library


```python
#https://www.kaggle.com/javadzabihi/happiness-2017-visualization-prediction/report

#The following command imports the CSV dataset using pandas:
happyness_2017 = pd.read_csv("happyness_2017.csv")

df=happyness_2017
#df
df.head() 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Happiness.Rank</th>
      <th>Happiness.Score</th>
      <th>Whisker.high</th>
      <th>Whisker.low</th>
      <th>Economy..GDP.per.Capita.</th>
      <th>Family</th>
      <th>Health..Life.Expectancy.</th>
      <th>Freedom</th>
      <th>Generosity</th>
      <th>Trust..Government.Corruption.</th>
      <th>Dystopia.Residual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Norway</td>
      <td>1</td>
      <td>7.537</td>
      <td>7.594445</td>
      <td>7.479556</td>
      <td>1.616463</td>
      <td>1.533524</td>
      <td>0.796667</td>
      <td>0.635423</td>
      <td>0.362012</td>
      <td>0.315964</td>
      <td>2.277027</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Denmark</td>
      <td>2</td>
      <td>7.522</td>
      <td>7.581728</td>
      <td>7.462272</td>
      <td>1.482383</td>
      <td>1.551122</td>
      <td>0.792566</td>
      <td>0.626007</td>
      <td>0.355280</td>
      <td>0.400770</td>
      <td>2.313707</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Iceland</td>
      <td>3</td>
      <td>7.504</td>
      <td>7.622030</td>
      <td>7.385970</td>
      <td>1.480633</td>
      <td>1.610574</td>
      <td>0.833552</td>
      <td>0.627163</td>
      <td>0.475540</td>
      <td>0.153527</td>
      <td>2.322715</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Switzerland</td>
      <td>4</td>
      <td>7.494</td>
      <td>7.561772</td>
      <td>7.426227</td>
      <td>1.564980</td>
      <td>1.516912</td>
      <td>0.858131</td>
      <td>0.620071</td>
      <td>0.290549</td>
      <td>0.367007</td>
      <td>2.276716</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Finland</td>
      <td>5</td>
      <td>7.469</td>
      <td>7.527542</td>
      <td>7.410458</td>
      <td>1.443572</td>
      <td>1.540247</td>
      <td>0.809158</td>
      <td>0.617951</td>
      <td>0.245483</td>
      <td>0.382612</td>
      <td>2.430182</td>
    </tr>
  </tbody>
</table>
</div>



**Looking at the current shape of the dataset under consideration**


```python
# Looking at the current shape of the dataset under consideration
#df.shape   

# Step 2:  check the dimension of the table or the size of dataframe

print("The dimension of the table is: ", df.shape)
```

    The dimension of the table is:  (155, 12)
    

### Cleaning - Is threre any missing  or null Values in this dataset (happyness_2017)?

In this section, we load our dataset and see the structure of happiness variables. Our dataset is pretty clean, and we will implement a few adjustments to make it looks better.


```python
#check for any missing values or null values (NA or NaN)
df.isnull().sum()
#df.isnull().head(6)
```




    Country                          0
    Happiness.Rank                   0
    Happiness.Score                  0
    Whisker.high                     0
    Whisker.low                      0
    Economy..GDP.per.Capita.         0
    Family                           0
    Health..Life.Expectancy.         0
    Freedom                          0
    Generosity                       0
    Trust..Government.Corruption.    0
    Dystopia.Residual                0
    dtype: int64



** Note that the above result no missing values so, the dataset is pretty cleaned.**


```python
# Print a list datatypes of all columns 
df.dtypes
```




    Country                           object
    Happiness.Rank                     int64
    Happiness.Score                  float64
    Whisker.high                     float64
    Whisker.low                      float64
    Economy..GDP.per.Capita.         float64
    Family                           float64
    Health..Life.Expectancy.         float64
    Freedom                          float64
    Generosity                       float64
    Trust..Government.Corruption.    float64
    Dystopia.Residual                float64
    dtype: object



### Exploratory Data Analysis

**Prints information of all columns:**


```python
df.info() # Prints information of all columns:
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 155 entries, 0 to 154
    Data columns (total 12 columns):
     #   Column                         Non-Null Count  Dtype  
    ---  ------                         --------------  -----  
     0   Country                        155 non-null    object 
     1   Happiness.Rank                 155 non-null    int64  
     2   Happiness.Score                155 non-null    float64
     3   Whisker.high                   155 non-null    float64
     4   Whisker.low                    155 non-null    float64
     5   Economy..GDP.per.Capita.       155 non-null    float64
     6   Family                         155 non-null    float64
     7   Health..Life.Expectancy.       155 non-null    float64
     8   Freedom                        155 non-null    float64
     9   Generosity                     155 non-null    float64
     10  Trust..Government.Corruption.  155 non-null    float64
     11  Dystopia.Residual              155 non-null    float64
    dtypes: float64(10), int64(1), object(1)
    memory usage: 14.7+ KB
    

**Display some statistical summaries of the numerical columns data. 
 To see the statistical details of the dataset, we can use describe():**


```python
df.describe().head()     # display some statistical summaries of the numerical columns data.
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Happiness.Rank</th>
      <th>Happiness.Score</th>
      <th>Whisker.high</th>
      <th>Whisker.low</th>
      <th>Economy..GDP.per.Capita.</th>
      <th>Family</th>
      <th>Health..Life.Expectancy.</th>
      <th>Freedom</th>
      <th>Generosity</th>
      <th>Trust..Government.Corruption.</th>
      <th>Dystopia.Residual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>155.000000</td>
      <td>155.000000</td>
      <td>155.000000</td>
      <td>155.000000</td>
      <td>155.000000</td>
      <td>155.000000</td>
      <td>155.000000</td>
      <td>155.000000</td>
      <td>155.000000</td>
      <td>155.000000</td>
      <td>155.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>78.000000</td>
      <td>5.354019</td>
      <td>5.452326</td>
      <td>5.255713</td>
      <td>0.984718</td>
      <td>1.188898</td>
      <td>0.551341</td>
      <td>0.408786</td>
      <td>0.246883</td>
      <td>0.123120</td>
      <td>1.850238</td>
    </tr>
    <tr>
      <th>std</th>
      <td>44.888751</td>
      <td>1.131230</td>
      <td>1.118542</td>
      <td>1.145030</td>
      <td>0.420793</td>
      <td>0.287263</td>
      <td>0.237073</td>
      <td>0.149997</td>
      <td>0.134780</td>
      <td>0.101661</td>
      <td>0.500028</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>2.693000</td>
      <td>2.864884</td>
      <td>2.521116</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.377914</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>39.500000</td>
      <td>4.505500</td>
      <td>4.608172</td>
      <td>4.374955</td>
      <td>0.663371</td>
      <td>1.042635</td>
      <td>0.369866</td>
      <td>0.303677</td>
      <td>0.154106</td>
      <td>0.057271</td>
      <td>1.591291</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.columns               # display the list of the columns
```




    Index(['Country', 'Happiness.Rank', 'Happiness.Score', 'Whisker.high',
           'Whisker.low', 'Economy..GDP.per.Capita.', 'Family',
           'Health..Life.Expectancy.', 'Freedom', 'Generosity',
           'Trust..Government.Corruption.', 'Dystopia.Residual'],
          dtype='object')



**Changing the name of columns**



```python
# To Changing the name of columns

df.columns=["Country", "Happiness.Rank", "Happiness.Score",
                          "Whisker.High", "Whisker.Low", "Economy", "Family",
                          "Life.Expectancy", "Freedom", "Generosity",
                          "Trust", "Dystopia.Residual"]

df.columns
```




    Index(['Country', 'Happiness.Rank', 'Happiness.Score', 'Whisker.High',
           'Whisker.Low', 'Economy', 'Family', 'Life.Expectancy', 'Freedom',
           'Generosity', 'Trust', 'Dystopia.Residual'],
          dtype='object')



**Removing unnecessary columns (Whisker.high and Whisker.low)**


```python
''' drop multiple column based on name in pandas'''

df_new = df.drop(['Whisker.High', 'Whisker.Low'], axis = 1)
df_new
df_new.shape
```




    (155, 10)




```python
df_new.columns
```




    Index(['Country', 'Happiness.Rank', 'Happiness.Score', 'Economy', 'Family',
           'Life.Expectancy', 'Freedom', 'Generosity', 'Trust',
           'Dystopia.Residual'],
          dtype='object')




```python
df_new
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Happiness.Rank</th>
      <th>Happiness.Score</th>
      <th>Economy</th>
      <th>Family</th>
      <th>Life.Expectancy</th>
      <th>Freedom</th>
      <th>Generosity</th>
      <th>Trust</th>
      <th>Dystopia.Residual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Norway</td>
      <td>1</td>
      <td>7.537</td>
      <td>1.616463</td>
      <td>1.533524</td>
      <td>0.796667</td>
      <td>0.635423</td>
      <td>0.362012</td>
      <td>0.315964</td>
      <td>2.277027</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Denmark</td>
      <td>2</td>
      <td>7.522</td>
      <td>1.482383</td>
      <td>1.551122</td>
      <td>0.792566</td>
      <td>0.626007</td>
      <td>0.355280</td>
      <td>0.400770</td>
      <td>2.313707</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Iceland</td>
      <td>3</td>
      <td>7.504</td>
      <td>1.480633</td>
      <td>1.610574</td>
      <td>0.833552</td>
      <td>0.627163</td>
      <td>0.475540</td>
      <td>0.153527</td>
      <td>2.322715</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Switzerland</td>
      <td>4</td>
      <td>7.494</td>
      <td>1.564980</td>
      <td>1.516912</td>
      <td>0.858131</td>
      <td>0.620071</td>
      <td>0.290549</td>
      <td>0.367007</td>
      <td>2.276716</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Finland</td>
      <td>5</td>
      <td>7.469</td>
      <td>1.443572</td>
      <td>1.540247</td>
      <td>0.809158</td>
      <td>0.617951</td>
      <td>0.245483</td>
      <td>0.382612</td>
      <td>2.430182</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>150</th>
      <td>Rwanda</td>
      <td>151</td>
      <td>3.471</td>
      <td>0.368746</td>
      <td>0.945707</td>
      <td>0.326425</td>
      <td>0.581844</td>
      <td>0.252756</td>
      <td>0.455220</td>
      <td>0.540061</td>
    </tr>
    <tr>
      <th>151</th>
      <td>Syria</td>
      <td>152</td>
      <td>3.462</td>
      <td>0.777153</td>
      <td>0.396103</td>
      <td>0.500533</td>
      <td>0.081539</td>
      <td>0.493664</td>
      <td>0.151347</td>
      <td>1.061574</td>
    </tr>
    <tr>
      <th>152</th>
      <td>Tanzania</td>
      <td>153</td>
      <td>3.349</td>
      <td>0.511136</td>
      <td>1.041990</td>
      <td>0.364509</td>
      <td>0.390018</td>
      <td>0.354256</td>
      <td>0.066035</td>
      <td>0.621130</td>
    </tr>
    <tr>
      <th>153</th>
      <td>Burundi</td>
      <td>154</td>
      <td>2.905</td>
      <td>0.091623</td>
      <td>0.629794</td>
      <td>0.151611</td>
      <td>0.059901</td>
      <td>0.204435</td>
      <td>0.084148</td>
      <td>1.683024</td>
    </tr>
    <tr>
      <th>154</th>
      <td>Central African Republic</td>
      <td>155</td>
      <td>2.693</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.018773</td>
      <td>0.270842</td>
      <td>0.280876</td>
      <td>0.056565</td>
      <td>2.066005</td>
    </tr>
  </tbody>
</table>
<p>155 rows × 10 columns</p>
</div>



### Visualization

### The correlation of the entire dataset 


```python
fig, ax = plt.subplots()
fig.set_size_inches(15, 10)
sns.heatmap(df.corr(),cmap='coolwarm',ax=ax,annot=True,linewidths=2)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x190e97f7be0>




![png](output_26_1.png)


Obviously, there is an inverse correlation between “Happiness Rank” and all the other numerical variables. In other words, the lower the happiness rank, the higher the happiness score, and the higher the other seven factors that contribute to happiness. So let’s remove the happiness rank, and see the correlation again.




### The correlation of the new dataset



```python
#The correlation of the new dataset
fig, ax = plt.subplots()
fig.set_size_inches(15, 10)
sns.heatmap(df_new.corr(),cmap='coolwarm',ax=ax,annot=True,linewidths=2)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x190e9d95ca0>




![png](output_29_1.png)


According to the above correlation plot, Economy, life expectancy, and family play the most significant role in contributing to happiness. Trust and generosity have the lowest impact on the happiness score.

***Using the histogram helps us to make the decision making process a lot more easy to handle by viewing the data that was collected***


```python
df_new.hist()
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x00000190EA478250>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x00000190EA6D3760>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x00000190EA700BE0>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x00000190EA72D0D0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x00000190EA7674F0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x00000190EA7918B0>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x00000190EA7919A0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x00000190EA7BCE50>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x00000190EA8236D0>]],
          dtype=object)




![png](output_32_1.png)



```python
sns.distplot(df['Happiness.Score'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x190ea7101c0>




![png](output_33_1.png)



```python
df_new.columns
```




    Index(['Country', 'Happiness.Rank', 'Happiness.Score', 'Economy', 'Family',
           'Life.Expectancy', 'Freedom', 'Generosity', 'Trust',
           'Dystopia.Residual'],
          dtype='object')



### Prediction- Setting up Linear Model to Predict Happiness

The following step allows to divide the data into attributes and labels. Attributes are the independent variables (X) while labels are dependent variables(y) whose values are to be predicted. In the new dataset, there are only have ten columns. We want to predict the happiness score depending upon the X recorded. Therefore, the attribute set consists of happiness. The score column, which is in the X variable, and the label will be the seven columns which is stored in the y variable.


In this section, we will implement several machine learning algorithms to predict happiness score. First, we should split our dataset into training and test set. The dependent variable is happiness score, and the independent variables are economy, family, life expectancy,freedom, generosity, trust, and dystopia residual.



```python
#X = df['attend'].values.reshape(-1,1)
#y = df['temp'].values.reshape(-1,1)

X = df_new.drop(['Happiness.Score', 'Happiness.Rank', 'Country'], axis=1)
#X = df_new.drop(['Happiness.Score', 'Happiness.Rank'], axis=1)
y = df_new['Happiness.Score']
X.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Economy</th>
      <th>Family</th>
      <th>Life.Expectancy</th>
      <th>Freedom</th>
      <th>Generosity</th>
      <th>Trust</th>
      <th>Dystopia.Residual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.616463</td>
      <td>1.533524</td>
      <td>0.796667</td>
      <td>0.635423</td>
      <td>0.362012</td>
      <td>0.315964</td>
      <td>2.277027</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.482383</td>
      <td>1.551122</td>
      <td>0.792566</td>
      <td>0.626007</td>
      <td>0.355280</td>
      <td>0.400770</td>
      <td>2.313707</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.480633</td>
      <td>1.610574</td>
      <td>0.833552</td>
      <td>0.627163</td>
      <td>0.475540</td>
      <td>0.153527</td>
      <td>2.322715</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.564980</td>
      <td>1.516912</td>
      <td>0.858131</td>
      <td>0.620071</td>
      <td>0.290549</td>
      <td>0.367007</td>
      <td>2.276716</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.443572</td>
      <td>1.540247</td>
      <td>0.809158</td>
      <td>0.617951</td>
      <td>0.245483</td>
      <td>0.382612</td>
      <td>2.430182</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Let's convert all the categorical variables into dummy variables
df = pd.get_dummies(df)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Happiness.Rank</th>
      <th>Happiness.Score</th>
      <th>Whisker.High</th>
      <th>Whisker.Low</th>
      <th>Economy</th>
      <th>Family</th>
      <th>Life.Expectancy</th>
      <th>Freedom</th>
      <th>Generosity</th>
      <th>Trust</th>
      <th>...</th>
      <th>Country_United Arab Emirates</th>
      <th>Country_United Kingdom</th>
      <th>Country_United States</th>
      <th>Country_Uruguay</th>
      <th>Country_Uzbekistan</th>
      <th>Country_Venezuela</th>
      <th>Country_Vietnam</th>
      <th>Country_Yemen</th>
      <th>Country_Zambia</th>
      <th>Country_Zimbabwe</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>7.537</td>
      <td>7.594445</td>
      <td>7.479556</td>
      <td>1.616463</td>
      <td>1.533524</td>
      <td>0.796667</td>
      <td>0.635423</td>
      <td>0.362012</td>
      <td>0.315964</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>7.522</td>
      <td>7.581728</td>
      <td>7.462272</td>
      <td>1.482383</td>
      <td>1.551122</td>
      <td>0.792566</td>
      <td>0.626007</td>
      <td>0.355280</td>
      <td>0.400770</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>7.504</td>
      <td>7.622030</td>
      <td>7.385970</td>
      <td>1.480633</td>
      <td>1.610574</td>
      <td>0.833552</td>
      <td>0.627163</td>
      <td>0.475540</td>
      <td>0.153527</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>7.494</td>
      <td>7.561772</td>
      <td>7.426227</td>
      <td>1.564980</td>
      <td>1.516912</td>
      <td>0.858131</td>
      <td>0.620071</td>
      <td>0.290549</td>
      <td>0.367007</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>7.469</td>
      <td>7.527542</td>
      <td>7.410458</td>
      <td>1.443572</td>
      <td>1.540247</td>
      <td>0.809158</td>
      <td>0.617951</td>
      <td>0.245483</td>
      <td>0.382612</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 166 columns</p>
</div>



Next, we split 80% of the data to the training set while 20% of the data to test set using below code.
The test_size variable is where we actually specify the proportion of the test set.


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
```

After splitting the data into training and testing sets, finally, the time is to train our algorithm. For that, we need to import LinearRegression class, instantiate it, and call the fit() method along with our training data.

Note that: lm stands for linear model and is called model or regressor


```python
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train, y_train) #training the algorithm

#regressor = LinearRegression()  
#regressor.fit(X_train, y_train) #training the algorithm
```




    LinearRegression()



The linear regression model basically finds the best value for the intercept and slope, which results in a line that best fits the data. To see the value of the intercept and slope calculated by the linear regression algorithm for our dataset, execute the following code.


```python
#To retrieve the intercept:
print(lm.intercept_)#For retrieving the slope:
print(lm.coef_)
```

    0.00021834398875419936
    [1.0000158  0.99988359 1.00010937 1.00007047 1.00010167 0.99977243
     0.99993477]
    


```python
#print('Coefficients: \n', lm.coef_)
#lm.coef_
```

This means that for every one unit of change in X, the change in the y is about 0.00158% to 99.988359

### Prediction


Now that we have trained our algorithm, it’s time to make some predictions. To do so, we will use our test data and see how accurately our algorithm predicts the percentage score. To make predictions on the test data, execute the following script:


```python
predictions = lm.predict( X_test)
predictions
```




    array([5.26228745, 4.69487725, 4.49692683, 4.13868112, 6.42250499,
           5.27908846, 6.09756958, 5.17492782, 3.80821618, 4.028374  ,
           6.0836513 , 5.75835021, 6.89103942, 5.01067949, 5.6115555 ,
           6.40310136, 7.46917627, 7.52182076, 5.27284344, 5.23371025,
           3.79483561, 4.80526035, 4.64431666, 5.85034742, 4.82862178,
           6.42444498, 5.07384085, 5.96303476, 4.46005885, 5.15138853,
           4.29067616, 6.07147555, 5.49317722, 5.50004829, 5.83757062,
           5.00369496, 4.03215988, 6.57214101, 5.5693284 , 3.76657622,
           5.32432747, 5.22971336, 5.2274094 , 4.5497721 , 4.18047076,
           5.18248584, 6.00844881])




```python
lm.score(X_test, y_test)
```




    0.999999877525094



**Comparing the actual output values for X_test with the predicted values, execute the following script:**


```python
df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Actual</th>
      <th>Predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>80</th>
      <td>5.262</td>
      <td>5.262287</td>
    </tr>
    <tr>
      <th>106</th>
      <td>4.695</td>
      <td>4.694877</td>
    </tr>
    <tr>
      <th>116</th>
      <td>4.497</td>
      <td>4.496927</td>
    </tr>
    <tr>
      <th>129</th>
      <td>4.139</td>
      <td>4.138681</td>
    </tr>
    <tr>
      <th>32</th>
      <td>6.422</td>
      <td>6.422505</td>
    </tr>
  </tbody>
</table>
</div>



***Create the scatter plot ***


```python
plt.scatter(y_test,predictions)
plt.xlabel('X_Test')
plt.ylabel('Predicted Y')
```




    Text(0, 0.5, 'Predicted Y')




![png](output_53_1.png)


Let us figure out the RMSE.The root-mean-square deviation (RMSD) or root-mean-square error (RMSE) is a frequently used measure of the differences between values (sample or population values) predicted by a model or an estimator and the values observed. The RMSD represents the square root of the second sample moment of the differences between predicted values and observed values or the quadratic mean of these differences.


```python
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
```

    MAE: 0.00028048667529037623
    MSE: 1.0037684771818889e-07
    RMSE: 0.00031682305427192147
    

As a result, RMSE is always non-negative, and a value of 0 (rarely achieved in practice) would indicate a perfect fit to the data. In general, a lower RMSD is better than a higher one. However, comparisons across different types of data would be invalid because the measure is dependent on the scale of the numbers used.


```python
coeffecients = pd.DataFrame(lm.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
coeffecients
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Coeffecient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Economy</th>
      <td>1.000016</td>
    </tr>
    <tr>
      <th>Family</th>
      <td>0.999884</td>
    </tr>
    <tr>
      <th>Life.Expectancy</th>
      <td>1.000109</td>
    </tr>
    <tr>
      <th>Freedom</th>
      <td>1.000070</td>
    </tr>
    <tr>
      <th>Generosity</th>
      <td>1.000102</td>
    </tr>
    <tr>
      <th>Trust</th>
      <td>0.999772</td>
    </tr>
    <tr>
      <th>Dystopia.Residual</th>
      <td>0.999935</td>
    </tr>
  </tbody>
</table>
</div>



**The above result shows that there is a positive correlation. This indicates that when the predictor variable increases, the response variable will also increase.**

Ref:
In statistics, the sign of each coefficient indicates the direction of the relationship between a predictor variable and the response variable.
A positive sign indicates that as the predictor variable increases, the response variable also increases.
A negative sign indicates that as the predictor variable increases, the response variable decreases.
https://statisticsbyjim.com/glossary/regression-coefficient/

**In this below section we can visualize the comparison result as a bar graph using the following script :**

**Note: As the number of records is huge, for representation purpose I’m taking just 25 records.**


```python
df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
```


![png](output_60_0.png)


Though our model is not very precise, the predicted percentages are equal to the actual ones.

### End Project1-DSC680
