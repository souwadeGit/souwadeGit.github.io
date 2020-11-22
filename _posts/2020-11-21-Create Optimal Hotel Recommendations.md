---
title: "Create Optimal Hotel Recommendations"
date: 2020-11-21
tags: [data science, data wrangling, messy data]
header:
  image: "/images/perceptron/percept.jpg"
excerpt: "Data Science, Data Wrangling, Messy Data"
mathjax: "true"
---


### 5.3 Assignment: Create Optimal Hotel Recommendations

All online travel agencies are scrambling to meet the Artificial Intelligence driven personalization standard set by Amazon and Netflix. In addition, the world of online travel has become a highly competitive space where brands try to capture our attention (and wallet) with recommending, comparing, matching, and sharing. For this assignment, we aim to create the optimal hotel recommendations for Expedia’s users that are searching for a hotel to book. For this assignment, you need to predict which “hotel cluster” the user is likely to book, given his (or her) search details. In doing so, you should be able to demonstrate your ability to use four different algorithms (of your choice). The data set can be found at Kaggle: Expedia Hotel Recommendations. To get you started, I would suggest you use train.csv which captured the logs of user behavior and destinations.csv which contains information related to hotel reviews made by users. You are also required to write a one page summary of your approach in getting to your prediction methods. I expect you to use a combination of R and Python in your answer.

https://www.kaggle.com/c/expedia-hotel-recommendations/data?select=train.csv

https://www.kaggle.com/c/expedia-hotel-recommendations/data?select=destinations.csv

https://pandas-profiling.github.io/pandas-profiling/docs/master/index.html

#pip install github.com/pandas-profiling/pandas-profiling/archive/master.zip


### File descriptions - data set can be found at Kaggle:

1. destinations.csv - contains information related to hotel reviews made by users(hotel search latent attributes)
2. train.csv - the training set- captured the logs of user behavior 
3. test.csv - the test set
4. sample_submission.csv - a sample submission file in the correct format

### The approach

The given problem is interpreted as a 100 class classification problem, where the classes are the hotel clusters.

### Load libraries


```python
#pip install pandas-profiling
```


```python
# Standard libraryimport-Python program to plot a complex bar chart  
import pandas as pd 
import numpy as np
import pandas_profiling as pp
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Use to configure display of graph
%matplotlib inline 

#stop unnecessary warnings from printing to the screen
warnings.simplefilter('ignore')

# third party imports
from datetime import datetime
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
```

### 1.  Import  dataset : destinations.csv - hotel search latent attributes


```python
import pandas as pd
#The following command imports the CSV dataset using pandas:
test = pd.read_csv("test.csv", nrows =10000)

destination = pd.read_csv("destinations.csv", nrows =10000)
df = pd.read_csv("destinations.csv", nrows =10000)
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
      <th>srch_destination_id</th>
      <th>d1</th>
      <th>d2</th>
      <th>d3</th>
      <th>d4</th>
      <th>d5</th>
      <th>d6</th>
      <th>d7</th>
      <th>d8</th>
      <th>d9</th>
      <th>...</th>
      <th>d140</th>
      <th>d141</th>
      <th>d142</th>
      <th>d143</th>
      <th>d144</th>
      <th>d145</th>
      <th>d146</th>
      <th>d147</th>
      <th>d148</th>
      <th>d149</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>-2.198657</td>
      <td>-2.198657</td>
      <td>-2.198657</td>
      <td>-2.198657</td>
      <td>-2.198657</td>
      <td>-1.897627</td>
      <td>-2.198657</td>
      <td>-2.198657</td>
      <td>-1.897627</td>
      <td>...</td>
      <td>-2.198657</td>
      <td>-2.198657</td>
      <td>-2.198657</td>
      <td>-2.198657</td>
      <td>-2.198657</td>
      <td>-2.198657</td>
      <td>-2.198657</td>
      <td>-2.198657</td>
      <td>-2.198657</td>
      <td>-2.198657</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>-2.181690</td>
      <td>-2.181690</td>
      <td>-2.181690</td>
      <td>-2.082564</td>
      <td>-2.181690</td>
      <td>-2.165028</td>
      <td>-2.181690</td>
      <td>-2.181690</td>
      <td>-2.031597</td>
      <td>...</td>
      <td>-2.165028</td>
      <td>-2.181690</td>
      <td>-2.165028</td>
      <td>-2.181690</td>
      <td>-2.181690</td>
      <td>-2.165028</td>
      <td>-2.181690</td>
      <td>-2.181690</td>
      <td>-2.181690</td>
      <td>-2.181690</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>-2.183490</td>
      <td>-2.224164</td>
      <td>-2.224164</td>
      <td>-2.189562</td>
      <td>-2.105819</td>
      <td>-2.075407</td>
      <td>-2.224164</td>
      <td>-2.118483</td>
      <td>-2.140393</td>
      <td>...</td>
      <td>-2.224164</td>
      <td>-2.224164</td>
      <td>-2.196379</td>
      <td>-2.224164</td>
      <td>-2.192009</td>
      <td>-2.224164</td>
      <td>-2.224164</td>
      <td>-2.224164</td>
      <td>-2.224164</td>
      <td>-2.057548</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>-2.177409</td>
      <td>-2.177409</td>
      <td>-2.177409</td>
      <td>-2.177409</td>
      <td>-2.177409</td>
      <td>-2.115485</td>
      <td>-2.177409</td>
      <td>-2.177409</td>
      <td>-2.177409</td>
      <td>...</td>
      <td>-2.161081</td>
      <td>-2.177409</td>
      <td>-2.177409</td>
      <td>-2.177409</td>
      <td>-2.177409</td>
      <td>-2.177409</td>
      <td>-2.177409</td>
      <td>-2.177409</td>
      <td>-2.177409</td>
      <td>-2.177409</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>-2.189562</td>
      <td>-2.187783</td>
      <td>-2.194008</td>
      <td>-2.171153</td>
      <td>-2.152303</td>
      <td>-2.056618</td>
      <td>-2.194008</td>
      <td>-2.194008</td>
      <td>-2.145911</td>
      <td>...</td>
      <td>-2.187356</td>
      <td>-2.194008</td>
      <td>-2.191779</td>
      <td>-2.194008</td>
      <td>-2.194008</td>
      <td>-2.185161</td>
      <td>-2.194008</td>
      <td>-2.194008</td>
      <td>-2.194008</td>
      <td>-2.188037</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 150 columns</p>
</div>




```python
# Showing the statistical details of the dataset
df.describe()
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
      <th>srch_destination_id</th>
      <th>d1</th>
      <th>d2</th>
      <th>d3</th>
      <th>d4</th>
      <th>d5</th>
      <th>d6</th>
      <th>d7</th>
      <th>d8</th>
      <th>d9</th>
      <th>...</th>
      <th>d140</th>
      <th>d141</th>
      <th>d142</th>
      <th>d143</th>
      <th>d144</th>
      <th>d145</th>
      <th>d146</th>
      <th>d147</th>
      <th>d148</th>
      <th>d149</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>...</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5126.342700</td>
      <td>-2.185688</td>
      <td>-2.196879</td>
      <td>-2.201013</td>
      <td>-2.187428</td>
      <td>-2.150445</td>
      <td>-2.083379</td>
      <td>-2.197001</td>
      <td>-2.197264</td>
      <td>-2.119438</td>
      <td>...</td>
      <td>-2.199784</td>
      <td>-2.186854</td>
      <td>-2.196236</td>
      <td>-2.197772</td>
      <td>-2.189854</td>
      <td>-2.199422</td>
      <td>-2.195256</td>
      <td>-2.202158</td>
      <td>-2.201880</td>
      <td>-2.191511</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2963.231721</td>
      <td>0.035098</td>
      <td>0.034648</td>
      <td>0.032592</td>
      <td>0.039848</td>
      <td>0.068166</td>
      <td>0.111477</td>
      <td>0.033512</td>
      <td>0.032764</td>
      <td>0.170400</td>
      <td>...</td>
      <td>0.029505</td>
      <td>0.050570</td>
      <td>0.039497</td>
      <td>0.033872</td>
      <td>0.039065</td>
      <td>0.030459</td>
      <td>0.045737</td>
      <td>0.030618</td>
      <td>0.030526</td>
      <td>0.038893</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>-2.376577</td>
      <td>-2.454624</td>
      <td>-2.454624</td>
      <td>-2.454624</td>
      <td>-2.454624</td>
      <td>-2.344165</td>
      <td>-2.454624</td>
      <td>-2.454624</td>
      <td>-2.376577</td>
      <td>...</td>
      <td>-2.426125</td>
      <td>-2.454624</td>
      <td>-2.454624</td>
      <td>-2.440107</td>
      <td>-2.454624</td>
      <td>-2.426125</td>
      <td>-2.454624</td>
      <td>-2.454624</td>
      <td>-2.454624</td>
      <td>-2.454624</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2576.750000</td>
      <td>-2.200926</td>
      <td>-2.212192</td>
      <td>-2.216285</td>
      <td>-2.204907</td>
      <td>-2.186731</td>
      <td>-2.163875</td>
      <td>-2.211987</td>
      <td>-2.212163</td>
      <td>-2.191913</td>
      <td>...</td>
      <td>-2.214207</td>
      <td>-2.205536</td>
      <td>-2.211689</td>
      <td>-2.213140</td>
      <td>-2.205800</td>
      <td>-2.215074</td>
      <td>-2.212077</td>
      <td>-2.216883</td>
      <td>-2.216439</td>
      <td>-2.207482</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5101.500000</td>
      <td>-2.182481</td>
      <td>-2.189541</td>
      <td>-2.192493</td>
      <td>-2.184689</td>
      <td>-2.176371</td>
      <td>-2.121796</td>
      <td>-2.189021</td>
      <td>-2.189218</td>
      <td>-2.176881</td>
      <td>...</td>
      <td>-2.191305</td>
      <td>-2.184773</td>
      <td>-2.188889</td>
      <td>-2.190416</td>
      <td>-2.185155</td>
      <td>-2.191353</td>
      <td>-2.189595</td>
      <td>-2.193526</td>
      <td>-2.193105</td>
      <td>-2.186003</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7686.250000</td>
      <td>-2.174647</td>
      <td>-2.177883</td>
      <td>-2.178964</td>
      <td>-2.175670</td>
      <td>-2.120533</td>
      <td>-2.036471</td>
      <td>-2.177730</td>
      <td>-2.177755</td>
      <td>-2.137957</td>
      <td>...</td>
      <td>-2.178139</td>
      <td>-2.175747</td>
      <td>-2.177680</td>
      <td>-2.178174</td>
      <td>-2.176011</td>
      <td>-2.177927</td>
      <td>-2.177703</td>
      <td>-2.179464</td>
      <td>-2.179312</td>
      <td>-2.176415</td>
    </tr>
    <tr>
      <th>max</th>
      <td>10326.000000</td>
      <td>-1.851415</td>
      <td>-1.586439</td>
      <td>-1.965178</td>
      <td>-1.936663</td>
      <td>-1.726651</td>
      <td>-1.209058</td>
      <td>-1.720070</td>
      <td>-1.879678</td>
      <td>-1.028502</td>
      <td>...</td>
      <td>-1.913814</td>
      <td>-0.987334</td>
      <td>-1.382385</td>
      <td>-1.775218</td>
      <td>-1.828735</td>
      <td>-1.838849</td>
      <td>-1.408689</td>
      <td>-1.942067</td>
      <td>-1.994061</td>
      <td>-1.717832</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 150 columns</p>
</div>



### Data Exploration - EDA ( Exploratory Data Analysis)


```python
# Data Exploration, shows the correlations

df_correlation = df.corr()

df_correlation
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
      <th>srch_destination_id</th>
      <th>d1</th>
      <th>d2</th>
      <th>d3</th>
      <th>d4</th>
      <th>d5</th>
      <th>d6</th>
      <th>d7</th>
      <th>d8</th>
      <th>d9</th>
      <th>...</th>
      <th>d140</th>
      <th>d141</th>
      <th>d142</th>
      <th>d143</th>
      <th>d144</th>
      <th>d145</th>
      <th>d146</th>
      <th>d147</th>
      <th>d148</th>
      <th>d149</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>srch_destination_id</th>
      <td>1.000000</td>
      <td>0.023452</td>
      <td>0.056412</td>
      <td>0.046895</td>
      <td>-0.025826</td>
      <td>-0.108487</td>
      <td>-0.079539</td>
      <td>0.040099</td>
      <td>0.061330</td>
      <td>0.001007</td>
      <td>...</td>
      <td>0.075543</td>
      <td>0.007303</td>
      <td>0.053665</td>
      <td>0.068324</td>
      <td>0.031330</td>
      <td>0.072233</td>
      <td>0.064384</td>
      <td>0.085488</td>
      <td>0.094296</td>
      <td>0.021082</td>
    </tr>
    <tr>
      <th>d1</th>
      <td>0.023452</td>
      <td>1.000000</td>
      <td>0.245350</td>
      <td>0.339934</td>
      <td>0.123480</td>
      <td>-0.024022</td>
      <td>-0.394337</td>
      <td>0.244634</td>
      <td>0.276315</td>
      <td>-0.091256</td>
      <td>...</td>
      <td>0.386723</td>
      <td>0.254219</td>
      <td>0.298462</td>
      <td>0.350614</td>
      <td>0.366729</td>
      <td>0.391173</td>
      <td>0.227590</td>
      <td>0.378712</td>
      <td>0.431959</td>
      <td>0.223985</td>
    </tr>
    <tr>
      <th>d2</th>
      <td>0.056412</td>
      <td>0.245350</td>
      <td>1.000000</td>
      <td>0.561313</td>
      <td>0.307271</td>
      <td>-0.067935</td>
      <td>-0.466595</td>
      <td>0.524542</td>
      <td>0.517061</td>
      <td>-0.256479</td>
      <td>...</td>
      <td>0.573922</td>
      <td>0.187761</td>
      <td>0.419645</td>
      <td>0.531607</td>
      <td>0.365113</td>
      <td>0.548796</td>
      <td>0.323933</td>
      <td>0.591859</td>
      <td>0.591054</td>
      <td>0.424194</td>
    </tr>
    <tr>
      <th>d3</th>
      <td>0.046895</td>
      <td>0.339934</td>
      <td>0.561313</td>
      <td>1.000000</td>
      <td>0.342952</td>
      <td>-0.209527</td>
      <td>-0.564215</td>
      <td>0.607864</td>
      <td>0.632385</td>
      <td>-0.354626</td>
      <td>...</td>
      <td>0.812518</td>
      <td>0.249292</td>
      <td>0.526792</td>
      <td>0.593746</td>
      <td>0.427997</td>
      <td>0.781456</td>
      <td>0.451202</td>
      <td>0.831343</td>
      <td>0.825003</td>
      <td>0.460593</td>
    </tr>
    <tr>
      <th>d4</th>
      <td>-0.025826</td>
      <td>0.123480</td>
      <td>0.307271</td>
      <td>0.342952</td>
      <td>1.000000</td>
      <td>0.264402</td>
      <td>-0.275328</td>
      <td>0.330128</td>
      <td>0.335002</td>
      <td>-0.273355</td>
      <td>...</td>
      <td>0.274738</td>
      <td>0.097304</td>
      <td>0.220406</td>
      <td>0.204212</td>
      <td>0.216866</td>
      <td>0.250476</td>
      <td>0.184501</td>
      <td>0.319789</td>
      <td>0.310443</td>
      <td>0.304064</td>
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
      <td>...</td>
    </tr>
    <tr>
      <th>d145</th>
      <td>0.072233</td>
      <td>0.391173</td>
      <td>0.548796</td>
      <td>0.781456</td>
      <td>0.250476</td>
      <td>-0.333771</td>
      <td>-0.539632</td>
      <td>0.577615</td>
      <td>0.612988</td>
      <td>-0.195059</td>
      <td>...</td>
      <td>0.904504</td>
      <td>0.222932</td>
      <td>0.519398</td>
      <td>0.641595</td>
      <td>0.387807</td>
      <td>1.000000</td>
      <td>0.451301</td>
      <td>0.836132</td>
      <td>0.829137</td>
      <td>0.400698</td>
    </tr>
    <tr>
      <th>d146</th>
      <td>0.064384</td>
      <td>0.227590</td>
      <td>0.323933</td>
      <td>0.451202</td>
      <td>0.184501</td>
      <td>-0.094769</td>
      <td>-0.401163</td>
      <td>0.357104</td>
      <td>0.367892</td>
      <td>-0.255916</td>
      <td>...</td>
      <td>0.463584</td>
      <td>0.209860</td>
      <td>0.305198</td>
      <td>0.362006</td>
      <td>0.285435</td>
      <td>0.451301</td>
      <td>1.000000</td>
      <td>0.492775</td>
      <td>0.484072</td>
      <td>0.247309</td>
    </tr>
    <tr>
      <th>d147</th>
      <td>0.085488</td>
      <td>0.378712</td>
      <td>0.591859</td>
      <td>0.831343</td>
      <td>0.319789</td>
      <td>-0.277476</td>
      <td>-0.603220</td>
      <td>0.647840</td>
      <td>0.682219</td>
      <td>-0.354815</td>
      <td>...</td>
      <td>0.866587</td>
      <td>0.276469</td>
      <td>0.557257</td>
      <td>0.657279</td>
      <td>0.448173</td>
      <td>0.836132</td>
      <td>0.492775</td>
      <td>1.000000</td>
      <td>0.887792</td>
      <td>0.466562</td>
    </tr>
    <tr>
      <th>d148</th>
      <td>0.094296</td>
      <td>0.431959</td>
      <td>0.591054</td>
      <td>0.825003</td>
      <td>0.310443</td>
      <td>-0.280455</td>
      <td>-0.616597</td>
      <td>0.637277</td>
      <td>0.669848</td>
      <td>-0.357010</td>
      <td>...</td>
      <td>0.866090</td>
      <td>0.296012</td>
      <td>0.570078</td>
      <td>0.642381</td>
      <td>0.473747</td>
      <td>0.829137</td>
      <td>0.484072</td>
      <td>0.887792</td>
      <td>1.000000</td>
      <td>0.461923</td>
    </tr>
    <tr>
      <th>d149</th>
      <td>0.021082</td>
      <td>0.223985</td>
      <td>0.424194</td>
      <td>0.460593</td>
      <td>0.304064</td>
      <td>0.077466</td>
      <td>-0.385431</td>
      <td>0.477479</td>
      <td>0.453188</td>
      <td>-0.295893</td>
      <td>...</td>
      <td>0.433573</td>
      <td>0.170423</td>
      <td>0.364597</td>
      <td>0.325766</td>
      <td>0.372327</td>
      <td>0.400698</td>
      <td>0.247309</td>
      <td>0.466562</td>
      <td>0.461923</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>150 rows × 150 columns</p>
</div>




```python
#Let’s explore the data a little bit by checking the number of rows and columns in our datasets.
df.shape
```




    (10000, 150)




```python
# display information
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10000 entries, 0 to 9999
    Columns: 150 entries, srch_destination_id to d149
    dtypes: float64(149), int64(1)
    memory usage: 11.4 MB
    


```python
#show columns
df.columns
```




    Index(['srch_destination_id', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8',
           'd9',
           ...
           'd140', 'd141', 'd142', 'd143', 'd144', 'd145', 'd146', 'd147', 'd148',
           'd149'],
          dtype='object', length=150)




```python
# How to identify the null value NaN where the value is equal to 0

#df.notnull().head()
df.notnull().sum()
```




    srch_destination_id    10000
    d1                     10000
    d2                     10000
    d3                     10000
    d4                     10000
                           ...  
    d145                   10000
    d146                   10000
    d147                   10000
    d148                   10000
    d149                   10000
    Length: 150, dtype: int64



The above line shows that there is no missing or null values


```python
# Histogram of the destinations file
df.hist('srch_destination_id')
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x000001B715D45AC0>]],
          dtype=object)




![png](output_15_1.png)



```python
import seaborn as seabornInstance 
plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(df['srch_destination_id'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1b715653490>




![png](output_16_1.png)



```python

#https://www.bing.com/videos/search?q=how+to+install+pandas_profiling+in+windows+10&docid=608012226248246912&mid=834EC20978002515E129834EC20978002515E129&view=detail&FORM=VIRE

#pip install pandas-profiling  

# pip install https://github.com/pandas-profiling/pandas-profiling/archive/master.zip


```


```python
#pip install pandas-profiling
import pandas as pd
import numpy as np
import pandas_profiling as pp
from pandas_profiling import ProfileReport
#df = pd.read_csv("destinations.csv", nrows= 10)
#df.head()
```


```python
#To generate the report
#profile = ProfileReport(df, title="Pandas Profiling Report")
#profile 
```


```python
# EDA of the train dataset
#profile = ProfileReport(df, title='Pandas Profiling Report', explorative=True)
```


```python
##This is achieved by simply displaying the report
#profile.to_widgets()
```

### 2.  Import  dataset :  train.csv - the training set


```python
import pandas as pd
#The following command imports the CSV dataset using pandas:

train = pd.read_csv("train.csv", nrows= 10000)
train.head() 
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
      <th>date_time</th>
      <th>site_name</th>
      <th>posa_continent</th>
      <th>user_location_country</th>
      <th>user_location_region</th>
      <th>user_location_city</th>
      <th>orig_destination_distance</th>
      <th>user_id</th>
      <th>is_mobile</th>
      <th>is_package</th>
      <th>...</th>
      <th>srch_children_cnt</th>
      <th>srch_rm_cnt</th>
      <th>srch_destination_id</th>
      <th>srch_destination_type_id</th>
      <th>is_booking</th>
      <th>cnt</th>
      <th>hotel_continent</th>
      <th>hotel_country</th>
      <th>hotel_market</th>
      <th>hotel_cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014-08-11 07:46:59</td>
      <td>2</td>
      <td>3</td>
      <td>66</td>
      <td>348</td>
      <td>48862</td>
      <td>2234.2641</td>
      <td>12</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>8250</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>50</td>
      <td>628</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014-08-11 08:22:12</td>
      <td>2</td>
      <td>3</td>
      <td>66</td>
      <td>348</td>
      <td>48862</td>
      <td>2234.2641</td>
      <td>12</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>8250</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>50</td>
      <td>628</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014-08-11 08:24:33</td>
      <td>2</td>
      <td>3</td>
      <td>66</td>
      <td>348</td>
      <td>48862</td>
      <td>2234.2641</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>8250</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>50</td>
      <td>628</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2014-08-09 18:05:16</td>
      <td>2</td>
      <td>3</td>
      <td>66</td>
      <td>442</td>
      <td>35390</td>
      <td>913.1932</td>
      <td>93</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>14984</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>50</td>
      <td>1457</td>
      <td>80</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014-08-09 18:08:18</td>
      <td>2</td>
      <td>3</td>
      <td>66</td>
      <td>442</td>
      <td>35390</td>
      <td>913.6259</td>
      <td>93</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>14984</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>50</td>
      <td>1457</td>
      <td>21</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>



### Exploratory Data Analysis (EDA)


```python
#pip install pandas-profiling
import pandas as pd
import numpy as np
import pandas_profiling as pp
from pandas_profiling import ProfileReport

#load 10 rows of the dataset
train = pd.read_csv("train.csv", nrows= 1000)
train.head()
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
      <th>date_time</th>
      <th>site_name</th>
      <th>posa_continent</th>
      <th>user_location_country</th>
      <th>user_location_region</th>
      <th>user_location_city</th>
      <th>orig_destination_distance</th>
      <th>user_id</th>
      <th>is_mobile</th>
      <th>is_package</th>
      <th>...</th>
      <th>srch_children_cnt</th>
      <th>srch_rm_cnt</th>
      <th>srch_destination_id</th>
      <th>srch_destination_type_id</th>
      <th>is_booking</th>
      <th>cnt</th>
      <th>hotel_continent</th>
      <th>hotel_country</th>
      <th>hotel_market</th>
      <th>hotel_cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014-08-11 07:46:59</td>
      <td>2</td>
      <td>3</td>
      <td>66</td>
      <td>348</td>
      <td>48862</td>
      <td>2234.2641</td>
      <td>12</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>8250</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>50</td>
      <td>628</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014-08-11 08:22:12</td>
      <td>2</td>
      <td>3</td>
      <td>66</td>
      <td>348</td>
      <td>48862</td>
      <td>2234.2641</td>
      <td>12</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>8250</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>50</td>
      <td>628</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014-08-11 08:24:33</td>
      <td>2</td>
      <td>3</td>
      <td>66</td>
      <td>348</td>
      <td>48862</td>
      <td>2234.2641</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>8250</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>50</td>
      <td>628</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2014-08-09 18:05:16</td>
      <td>2</td>
      <td>3</td>
      <td>66</td>
      <td>442</td>
      <td>35390</td>
      <td>913.1932</td>
      <td>93</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>14984</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>50</td>
      <td>1457</td>
      <td>80</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014-08-09 18:08:18</td>
      <td>2</td>
      <td>3</td>
      <td>66</td>
      <td>442</td>
      <td>35390</td>
      <td>913.6259</td>
      <td>93</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>14984</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>50</td>
      <td>1457</td>
      <td>21</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>




```python
#To generate the report
profile = ProfileReport(train, title="Pandas Profiling Report")
#profile 
```


```python
# EDA of the train dataset
#pp.ProfileReport(train)
profile = ProfileReport(train, title='Pandas Profiling Report', explorative=True)
```


```python
#This is achieved by simply displaying the report
#profile.to_widgets()
```


```python
profile = train.profile_report(title='Pandas Profiling Report', plot={'histogram': {'bins': 8}})
profile.to_file("output.html")
```


    HBox(children=(FloatProgress(value=0.0, description='Summarize dataset', max=38.0, style=ProgressStyle(descrip…


    
    


    HBox(children=(FloatProgress(value=0.0, description='Generate report structure', max=1.0, style=ProgressStyle(…


    
    


    HBox(children=(FloatProgress(value=0.0, description='Render HTML', max=1.0, style=ProgressStyle(description_wi…


    
    


    HBox(children=(FloatProgress(value=0.0, description='Export report to file', max=1.0, style=ProgressStyle(desc…


    
    


```python
# generate a HTML report file, save the ProfileReport to an object
profile.to_file("your_report.html")
```


    HBox(children=(FloatProgress(value=0.0, description='Export report to file', max=1.0, style=ProgressStyle(desc…


    
    

### Data Exploration


```python
# to drop column that contains missing or null values
clean_train = train.dropna(axis='columns')
clean_train.head() 
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
      <th>date_time</th>
      <th>site_name</th>
      <th>posa_continent</th>
      <th>user_location_country</th>
      <th>user_location_region</th>
      <th>user_location_city</th>
      <th>user_id</th>
      <th>is_mobile</th>
      <th>is_package</th>
      <th>channel</th>
      <th>...</th>
      <th>srch_children_cnt</th>
      <th>srch_rm_cnt</th>
      <th>srch_destination_id</th>
      <th>srch_destination_type_id</th>
      <th>is_booking</th>
      <th>cnt</th>
      <th>hotel_continent</th>
      <th>hotel_country</th>
      <th>hotel_market</th>
      <th>hotel_cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014-08-11 07:46:59</td>
      <td>2</td>
      <td>3</td>
      <td>66</td>
      <td>348</td>
      <td>48862</td>
      <td>12</td>
      <td>0</td>
      <td>1</td>
      <td>9</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>8250</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>50</td>
      <td>628</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014-08-11 08:22:12</td>
      <td>2</td>
      <td>3</td>
      <td>66</td>
      <td>348</td>
      <td>48862</td>
      <td>12</td>
      <td>0</td>
      <td>1</td>
      <td>9</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>8250</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>50</td>
      <td>628</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014-08-11 08:24:33</td>
      <td>2</td>
      <td>3</td>
      <td>66</td>
      <td>348</td>
      <td>48862</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>8250</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>50</td>
      <td>628</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2014-08-09 18:05:16</td>
      <td>2</td>
      <td>3</td>
      <td>66</td>
      <td>442</td>
      <td>35390</td>
      <td>93</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>14984</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>50</td>
      <td>1457</td>
      <td>80</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014-08-09 18:08:18</td>
      <td>2</td>
      <td>3</td>
      <td>66</td>
      <td>442</td>
      <td>35390</td>
      <td>93</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>14984</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>50</td>
      <td>1457</td>
      <td>21</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python
# Data Exploration, shows the correlations

train_correlation = train.corr()

#train_correlation
```


```python
#Let’s explore the data a little bit by checking the number of rows and columns in our datasets.
train.shape
```




    (1000, 24)




```python
# Showing the statistical details of the dataset
train.describe()
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
      <th>site_name</th>
      <th>posa_continent</th>
      <th>user_location_country</th>
      <th>user_location_region</th>
      <th>user_location_city</th>
      <th>orig_destination_distance</th>
      <th>user_id</th>
      <th>is_mobile</th>
      <th>is_package</th>
      <th>channel</th>
      <th>...</th>
      <th>srch_children_cnt</th>
      <th>srch_rm_cnt</th>
      <th>srch_destination_id</th>
      <th>srch_destination_type_id</th>
      <th>is_booking</th>
      <th>cnt</th>
      <th>hotel_continent</th>
      <th>hotel_country</th>
      <th>hotel_market</th>
      <th>hotel_cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1000.00000</td>
      <td>1000.00000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>268.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>...</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.00000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>19.36100</td>
      <td>2.16700</td>
      <td>50.865000</td>
      <td>193.805000</td>
      <td>19680.638000</td>
      <td>1860.755094</td>
      <td>3596.333000</td>
      <td>0.343000</td>
      <td>0.140000</td>
      <td>4.850000</td>
      <td>...</td>
      <td>0.353000</td>
      <td>1.110000</td>
      <td>15154.889000</td>
      <td>2.677000</td>
      <td>0.064000</td>
      <td>1.395000</td>
      <td>3.49400</td>
      <td>87.145000</td>
      <td>406.705000</td>
      <td>48.255000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>10.30577</td>
      <td>0.74274</td>
      <td>56.595334</td>
      <td>243.919765</td>
      <td>16541.209223</td>
      <td>2271.610410</td>
      <td>1499.094642</td>
      <td>0.474949</td>
      <td>0.347161</td>
      <td>3.533835</td>
      <td>...</td>
      <td>0.555608</td>
      <td>0.440561</td>
      <td>11817.903568</td>
      <td>2.296071</td>
      <td>0.244875</td>
      <td>1.159448</td>
      <td>1.82189</td>
      <td>50.001001</td>
      <td>404.375879</td>
      <td>29.048128</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2.00000</td>
      <td>0.00000</td>
      <td>3.000000</td>
      <td>12.000000</td>
      <td>1493.000000</td>
      <td>3.337900</td>
      <td>12.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>267.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>13.00000</td>
      <td>2.00000</td>
      <td>23.000000</td>
      <td>48.000000</td>
      <td>4924.000000</td>
      <td>177.330075</td>
      <td>2451.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>8278.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2.00000</td>
      <td>50.000000</td>
      <td>35.000000</td>
      <td>24.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>24.00000</td>
      <td>2.00000</td>
      <td>23.000000</td>
      <td>64.000000</td>
      <td>10067.000000</td>
      <td>766.156100</td>
      <td>3972.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>8811.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2.00000</td>
      <td>50.000000</td>
      <td>366.000000</td>
      <td>43.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>25.00000</td>
      <td>3.00000</td>
      <td>66.000000</td>
      <td>189.000000</td>
      <td>40365.000000</td>
      <td>2454.858800</td>
      <td>4539.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>9.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>18489.000000</td>
      <td>6.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>6.00000</td>
      <td>105.000000</td>
      <td>628.000000</td>
      <td>72.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>37.00000</td>
      <td>4.00000</td>
      <td>205.000000</td>
      <td>991.000000</td>
      <td>56440.000000</td>
      <td>8457.263600</td>
      <td>6450.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>9.000000</td>
      <td>...</td>
      <td>3.000000</td>
      <td>5.000000</td>
      <td>65035.000000</td>
      <td>8.000000</td>
      <td>1.000000</td>
      <td>23.000000</td>
      <td>6.00000</td>
      <td>208.000000</td>
      <td>1926.000000</td>
      <td>99.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 21 columns</p>
</div>




```python
#train.info()
```


```python
#show columns
train.columns
```




    Index(['date_time', 'site_name', 'posa_continent', 'user_location_country',
           'user_location_region', 'user_location_city',
           'orig_destination_distance', 'user_id', 'is_mobile', 'is_package',
           'channel', 'srch_ci', 'srch_co', 'srch_adults_cnt', 'srch_children_cnt',
           'srch_rm_cnt', 'srch_destination_id', 'srch_destination_type_id',
           'is_booking', 'cnt', 'hotel_continent', 'hotel_country', 'hotel_market',
           'hotel_cluster'],
          dtype='object')




```python
# How to identify the null value NaN where the value is equal to 0

#df.notnull().head()
train.notnull().sum()
```




    date_time                    1000
    site_name                    1000
    posa_continent               1000
    user_location_country        1000
    user_location_region         1000
    user_location_city           1000
    orig_destination_distance     268
    user_id                      1000
    is_mobile                    1000
    is_package                   1000
    channel                      1000
    srch_ci                      1000
    srch_co                      1000
    srch_adults_cnt              1000
    srch_children_cnt            1000
    srch_rm_cnt                  1000
    srch_destination_id          1000
    srch_destination_type_id     1000
    is_booking                   1000
    cnt                          1000
    hotel_continent              1000
    hotel_country                1000
    hotel_market                 1000
    hotel_cluster                1000
    dtype: int64




```python
# to drop column that contains missing or null values
clean_train = train.dropna(axis='columns')
clean_train.head() 
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
      <th>date_time</th>
      <th>site_name</th>
      <th>posa_continent</th>
      <th>user_location_country</th>
      <th>user_location_region</th>
      <th>user_location_city</th>
      <th>user_id</th>
      <th>is_mobile</th>
      <th>is_package</th>
      <th>channel</th>
      <th>...</th>
      <th>srch_children_cnt</th>
      <th>srch_rm_cnt</th>
      <th>srch_destination_id</th>
      <th>srch_destination_type_id</th>
      <th>is_booking</th>
      <th>cnt</th>
      <th>hotel_continent</th>
      <th>hotel_country</th>
      <th>hotel_market</th>
      <th>hotel_cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014-08-11 07:46:59</td>
      <td>2</td>
      <td>3</td>
      <td>66</td>
      <td>348</td>
      <td>48862</td>
      <td>12</td>
      <td>0</td>
      <td>1</td>
      <td>9</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>8250</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>50</td>
      <td>628</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014-08-11 08:22:12</td>
      <td>2</td>
      <td>3</td>
      <td>66</td>
      <td>348</td>
      <td>48862</td>
      <td>12</td>
      <td>0</td>
      <td>1</td>
      <td>9</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>8250</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>50</td>
      <td>628</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014-08-11 08:24:33</td>
      <td>2</td>
      <td>3</td>
      <td>66</td>
      <td>348</td>
      <td>48862</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>8250</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>50</td>
      <td>628</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2014-08-09 18:05:16</td>
      <td>2</td>
      <td>3</td>
      <td>66</td>
      <td>442</td>
      <td>35390</td>
      <td>93</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>14984</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>50</td>
      <td>1457</td>
      <td>80</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014-08-09 18:08:18</td>
      <td>2</td>
      <td>3</td>
      <td>66</td>
      <td>442</td>
      <td>35390</td>
      <td>93</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>14984</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>50</td>
      <td>1457</td>
      <td>21</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python
# Histogram of the train dataset 
train.hist('srch_destination_id')
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x000001B7156CB100>]],
          dtype=object)




![png](output_40_1.png)



```python
import matplotlib.pyplot as plt
import seaborn as sns
# histogram of clusters
plt.figure(figsize=(12, 6))
sns.distplot(train['hotel_cluster'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1b73c1a8790>




![png](output_41_1.png)


### The above histogram of hotel clusters display that the data is distributed over all 100 clusters. 


```python
#import seaborn as seabornInstance 
#plt.figure(figsize=(15,10))
#plt.tight_layout()
#seabornInstance.distplot(train['srch_destination_type_id'])
```

### Feature Engineering

In the train dataset, date columns can not be used directly in the model. Therefore it is necessary to extract year and month from them.

* **date_time** - Timestamp
* **srch_ci** - Checkin date
* **srch_co** - Checkout date


```python
train.head()
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
      <th>date_time</th>
      <th>site_name</th>
      <th>posa_continent</th>
      <th>user_location_country</th>
      <th>user_location_region</th>
      <th>user_location_city</th>
      <th>orig_destination_distance</th>
      <th>user_id</th>
      <th>is_mobile</th>
      <th>is_package</th>
      <th>...</th>
      <th>srch_children_cnt</th>
      <th>srch_rm_cnt</th>
      <th>srch_destination_id</th>
      <th>srch_destination_type_id</th>
      <th>is_booking</th>
      <th>cnt</th>
      <th>hotel_continent</th>
      <th>hotel_country</th>
      <th>hotel_market</th>
      <th>hotel_cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014-08-11 07:46:59</td>
      <td>2</td>
      <td>3</td>
      <td>66</td>
      <td>348</td>
      <td>48862</td>
      <td>2234.2641</td>
      <td>12</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>8250</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>50</td>
      <td>628</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014-08-11 08:22:12</td>
      <td>2</td>
      <td>3</td>
      <td>66</td>
      <td>348</td>
      <td>48862</td>
      <td>2234.2641</td>
      <td>12</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>8250</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>50</td>
      <td>628</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014-08-11 08:24:33</td>
      <td>2</td>
      <td>3</td>
      <td>66</td>
      <td>348</td>
      <td>48862</td>
      <td>2234.2641</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>8250</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>50</td>
      <td>628</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2014-08-09 18:05:16</td>
      <td>2</td>
      <td>3</td>
      <td>66</td>
      <td>442</td>
      <td>35390</td>
      <td>913.1932</td>
      <td>93</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>14984</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>50</td>
      <td>1457</td>
      <td>80</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014-08-09 18:08:18</td>
      <td>2</td>
      <td>3</td>
      <td>66</td>
      <td>442</td>
      <td>35390</td>
      <td>913.6259</td>
      <td>93</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>14984</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>50</td>
      <td>1457</td>
      <td>21</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>




```python
train.columns
```




    Index(['date_time', 'site_name', 'posa_continent', 'user_location_country',
           'user_location_region', 'user_location_city',
           'orig_destination_distance', 'user_id', 'is_mobile', 'is_package',
           'channel', 'srch_ci', 'srch_co', 'srch_adults_cnt', 'srch_children_cnt',
           'srch_rm_cnt', 'srch_destination_id', 'srch_destination_type_id',
           'is_booking', 'cnt', 'hotel_continent', 'hotel_country', 'hotel_market',
           'hotel_cluster'],
          dtype='object')




```python
# get year part from a date

def get_year(x):
    '''
    Args:
        datetime
    Returns:
        year as numeric
    '''
    if x is not None and type(x) is not float:
        try:
            return datetime.strptime(x, '%Y-%m-%d').year
        except ValueError:
            return datetime.strptime(x, '%Y-%m-%d %H:%M:%S').year
    else:
        return 2013
    pass

# get month part from a date

def get_month(x):
    '''
    Args:
        datetime
    Returns:
        month as numeric
    '''    
    if x is not None and type(x) is not float:
        try:
            return datetime.strptime(x, '%Y-%m-%d').month
        except:
            return datetime.strptime(x, '%Y-%m-%d %H:%M:%S').month
    else:
        return 1
    pass

# extract year and month from date time column
train['date_time_year'] = pd.Series(train.date_time, index = train.index)
train['date_time_month'] = pd.Series(train.date_time, index = train.index)

train.date_time_year = train.date_time_year.apply(lambda x: get_year(x))
train.date_time_month = train.date_time_month.apply(lambda x: get_month(x))
del train['date_time']

# extract year and month from check in date column
train['srch_ci_year'] = pd.Series(train.srch_ci, index = train.index)
train['srch_ci_month'] = pd.Series(train.srch_ci, index = train.index)

train.srch_ci_year = train.srch_ci_year.apply(lambda x: get_year(x))
train.srch_ci_month = train.srch_ci_month.apply(lambda x: get_month(x))
del train['srch_ci']

# extract year and month from check out date column
train['srch_co_year'] = pd.Series(train.srch_co, index = train.index)
train['srch_co_month'] = pd.Series(train.srch_co, index = train.index)

train.srch_co_year = train.srch_co_year.apply(lambda x: get_year(x))
train.srch_co_month = train.srch_co_month.apply(lambda x: get_month(x))
del train['srch_co']

# check the transformed data
train.head()
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
      <th>site_name</th>
      <th>posa_continent</th>
      <th>user_location_country</th>
      <th>user_location_region</th>
      <th>user_location_city</th>
      <th>orig_destination_distance</th>
      <th>user_id</th>
      <th>is_mobile</th>
      <th>is_package</th>
      <th>channel</th>
      <th>...</th>
      <th>hotel_continent</th>
      <th>hotel_country</th>
      <th>hotel_market</th>
      <th>hotel_cluster</th>
      <th>date_time_year</th>
      <th>date_time_month</th>
      <th>srch_ci_year</th>
      <th>srch_ci_month</th>
      <th>srch_co_year</th>
      <th>srch_co_month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>3</td>
      <td>66</td>
      <td>348</td>
      <td>48862</td>
      <td>2234.2641</td>
      <td>12</td>
      <td>0</td>
      <td>1</td>
      <td>9</td>
      <td>...</td>
      <td>2</td>
      <td>50</td>
      <td>628</td>
      <td>1</td>
      <td>2014</td>
      <td>8</td>
      <td>2014</td>
      <td>8</td>
      <td>2014</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>3</td>
      <td>66</td>
      <td>348</td>
      <td>48862</td>
      <td>2234.2641</td>
      <td>12</td>
      <td>0</td>
      <td>1</td>
      <td>9</td>
      <td>...</td>
      <td>2</td>
      <td>50</td>
      <td>628</td>
      <td>1</td>
      <td>2014</td>
      <td>8</td>
      <td>2014</td>
      <td>8</td>
      <td>2014</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>3</td>
      <td>66</td>
      <td>348</td>
      <td>48862</td>
      <td>2234.2641</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>...</td>
      <td>2</td>
      <td>50</td>
      <td>628</td>
      <td>1</td>
      <td>2014</td>
      <td>8</td>
      <td>2014</td>
      <td>8</td>
      <td>2014</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>3</td>
      <td>66</td>
      <td>442</td>
      <td>35390</td>
      <td>913.1932</td>
      <td>93</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>...</td>
      <td>2</td>
      <td>50</td>
      <td>1457</td>
      <td>80</td>
      <td>2014</td>
      <td>8</td>
      <td>2014</td>
      <td>11</td>
      <td>2014</td>
      <td>11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>3</td>
      <td>66</td>
      <td>442</td>
      <td>35390</td>
      <td>913.6259</td>
      <td>93</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>...</td>
      <td>2</td>
      <td>50</td>
      <td>1457</td>
      <td>21</td>
      <td>2014</td>
      <td>8</td>
      <td>2014</td>
      <td>11</td>
      <td>2014</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>



### The correlation of the entire dataset -train.csv


```python
fig, ax = plt.subplots()
fig.set_size_inches(15, 10)
sns.heatmap(train.corr(),cmap='coolwarm',ax=ax,annot=True,linewidths=2)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1b73d4b9a00>




![png](output_49_1.png)


###  The above graph  show the correlation between others variables with the hotel cluster 


```python
# correlation with others
train.corr()["hotel_cluster"].sort_values()
```




    date_time_month             -0.151771
    hotel_continent             -0.108342
    hotel_country               -0.091295
    site_name                   -0.072002
    srch_ci_year                -0.068858
    srch_co_year                -0.068650
    srch_destination_type_id    -0.061288
    srch_rm_cnt                 -0.037784
    srch_adults_cnt             -0.028777
    cnt                         -0.021956
    channel                     -0.013903
    srch_destination_id         -0.013032
    user_location_city           0.000234
    is_booking                   0.008258
    posa_continent               0.017371
    is_package                   0.028220
    srch_co_month                0.034409
    srch_ci_month                0.037463
    date_time_year               0.037519
    user_id                      0.041986
    user_location_country        0.045645
    user_location_region         0.053300
    srch_children_cnt            0.060347
    is_mobile                    0.067806
    hotel_market                 0.095300
    orig_destination_distance    0.104659
    hotel_cluster                1.000000
    Name: hotel_cluster, dtype: float64



The relationship( linear correlation) between the hotel cluster and other variables is not strong. The methods in which model linear relationship between features might not be suitable for the problem. The following factors will be impactful when it comes to clustering:

1. srch_destination_id - ID of the destination where the hotel search was performed
2. hotel_country - Country where the hotel is located
3. hotel_market - Hotel market
4. hotel_cluster - ID of a hotel cluster
5. is_package - Whether part of a package or not (1/0)
6. is_booking - Booking (1) or Click (0)


```python
#There is an interest in booking events,so let us get rid of clicks.
train_book = train.loc[train['is_booking'] == 1]
```

### Create a pivot to map each cluster, and shape it accordingly so that it can be merged with the original data.


```python
# step 1
factors = [train_book.groupby(['srch_destination_id','hotel_country','hotel_market','is_package','hotel_cluster'])['is_booking'].agg(['sum','count'])]
summ = pd.concat(factors).groupby(level=[0,1,2,3,4]).sum()
summ.dropna(inplace=True)
summ.head()
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
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th>sum</th>
      <th>count</th>
    </tr>
    <tr>
      <th>srch_destination_id</th>
      <th>hotel_country</th>
      <th>hotel_market</th>
      <th>is_package</th>
      <th>hotel_cluster</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1385</th>
      <th>185</th>
      <th>185</th>
      <th>1</th>
      <th>58</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1571</th>
      <th>5</th>
      <th>89</th>
      <th>0</th>
      <th>38</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4777</th>
      <th>50</th>
      <th>967</th>
      <th>0</th>
      <th>42</th>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5080</th>
      <th>204</th>
      <th>1762</th>
      <th>0</th>
      <th>61</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8213</th>
      <th>68</th>
      <th>275</th>
      <th>1</th>
      <th>68</th>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
# step 2
summ['sum_and_cnt'] = 0.85*summ['sum'] + 0.15*summ['count']
summ = summ.groupby(level=[0,1,2,3]).apply(lambda x: x.astype(float)/x.sum())
summ.reset_index(inplace=True)
summ.head()
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
      <th>srch_destination_id</th>
      <th>hotel_country</th>
      <th>hotel_market</th>
      <th>is_package</th>
      <th>hotel_cluster</th>
      <th>sum</th>
      <th>count</th>
      <th>sum_and_cnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1385</td>
      <td>185</td>
      <td>185</td>
      <td>1</td>
      <td>58</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1571</td>
      <td>5</td>
      <td>89</td>
      <td>0</td>
      <td>38</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4777</td>
      <td>50</td>
      <td>967</td>
      <td>0</td>
      <td>42</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5080</td>
      <td>204</td>
      <td>1762</td>
      <td>0</td>
      <td>61</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8213</td>
      <td>68</td>
      <td>275</td>
      <td>1</td>
      <td>68</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# step 3
summ_pivot = summ.pivot_table(index=['srch_destination_id','hotel_country','hotel_market','is_package'], columns='hotel_cluster', values='sum_and_cnt').reset_index()
summ_pivot.head()
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
      <th>hotel_cluster</th>
      <th>srch_destination_id</th>
      <th>hotel_country</th>
      <th>hotel_market</th>
      <th>is_package</th>
      <th>1</th>
      <th>2</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>10</th>
      <th>...</th>
      <th>78</th>
      <th>80</th>
      <th>81</th>
      <th>82</th>
      <th>83</th>
      <th>85</th>
      <th>90</th>
      <th>91</th>
      <th>95</th>
      <th>99</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1385</td>
      <td>185</td>
      <td>185</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1571</td>
      <td>5</td>
      <td>89</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4777</td>
      <td>50</td>
      <td>967</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5080</td>
      <td>204</td>
      <td>1762</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8213</td>
      <td>68</td>
      <td>275</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 48 columns</p>
</div>




```python
# check the destination data to determine the relationship with other data.
df = pd.read_csv("destinations.csv", nrows =100000)
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
      <th>srch_destination_id</th>
      <th>d1</th>
      <th>d2</th>
      <th>d3</th>
      <th>d4</th>
      <th>d5</th>
      <th>d6</th>
      <th>d7</th>
      <th>d8</th>
      <th>d9</th>
      <th>...</th>
      <th>d140</th>
      <th>d141</th>
      <th>d142</th>
      <th>d143</th>
      <th>d144</th>
      <th>d145</th>
      <th>d146</th>
      <th>d147</th>
      <th>d148</th>
      <th>d149</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>-2.198657</td>
      <td>-2.198657</td>
      <td>-2.198657</td>
      <td>-2.198657</td>
      <td>-2.198657</td>
      <td>-1.897627</td>
      <td>-2.198657</td>
      <td>-2.198657</td>
      <td>-1.897627</td>
      <td>...</td>
      <td>-2.198657</td>
      <td>-2.198657</td>
      <td>-2.198657</td>
      <td>-2.198657</td>
      <td>-2.198657</td>
      <td>-2.198657</td>
      <td>-2.198657</td>
      <td>-2.198657</td>
      <td>-2.198657</td>
      <td>-2.198657</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>-2.181690</td>
      <td>-2.181690</td>
      <td>-2.181690</td>
      <td>-2.082564</td>
      <td>-2.181690</td>
      <td>-2.165028</td>
      <td>-2.181690</td>
      <td>-2.181690</td>
      <td>-2.031597</td>
      <td>...</td>
      <td>-2.165028</td>
      <td>-2.181690</td>
      <td>-2.165028</td>
      <td>-2.181690</td>
      <td>-2.181690</td>
      <td>-2.165028</td>
      <td>-2.181690</td>
      <td>-2.181690</td>
      <td>-2.181690</td>
      <td>-2.181690</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>-2.183490</td>
      <td>-2.224164</td>
      <td>-2.224164</td>
      <td>-2.189562</td>
      <td>-2.105819</td>
      <td>-2.075407</td>
      <td>-2.224164</td>
      <td>-2.118483</td>
      <td>-2.140393</td>
      <td>...</td>
      <td>-2.224164</td>
      <td>-2.224164</td>
      <td>-2.196379</td>
      <td>-2.224164</td>
      <td>-2.192009</td>
      <td>-2.224164</td>
      <td>-2.224164</td>
      <td>-2.224164</td>
      <td>-2.224164</td>
      <td>-2.057548</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>-2.177409</td>
      <td>-2.177409</td>
      <td>-2.177409</td>
      <td>-2.177409</td>
      <td>-2.177409</td>
      <td>-2.115485</td>
      <td>-2.177409</td>
      <td>-2.177409</td>
      <td>-2.177409</td>
      <td>...</td>
      <td>-2.161081</td>
      <td>-2.177409</td>
      <td>-2.177409</td>
      <td>-2.177409</td>
      <td>-2.177409</td>
      <td>-2.177409</td>
      <td>-2.177409</td>
      <td>-2.177409</td>
      <td>-2.177409</td>
      <td>-2.177409</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>-2.189562</td>
      <td>-2.187783</td>
      <td>-2.194008</td>
      <td>-2.171153</td>
      <td>-2.152303</td>
      <td>-2.056618</td>
      <td>-2.194008</td>
      <td>-2.194008</td>
      <td>-2.145911</td>
      <td>...</td>
      <td>-2.187356</td>
      <td>-2.194008</td>
      <td>-2.191779</td>
      <td>-2.194008</td>
      <td>-2.194008</td>
      <td>-2.185161</td>
      <td>-2.194008</td>
      <td>-2.194008</td>
      <td>-2.194008</td>
      <td>-2.188037</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 150 columns</p>
</div>



### Merge the filtered booking data, pivotted data and destination data to form a single wide dataset.


```python
destination = pd.read_csv("destinations.csv", nrows=100000)
```


```python
train_book = pd.merge(train_book, destination, how='left', on='srch_destination_id')
train_book = pd.merge(train_book, summ_pivot, how='left', on=['srch_destination_id','hotel_country','hotel_market','is_package'])
train_book.fillna(0, inplace=True)
train_book.shape
```




    (64, 220)




```python
print(train_book.head())
```

       site_name  posa_continent  user_location_country  user_location_region  \
    0          2               3                     66                   348   
    1          2               3                     66                   318   
    2         30               4                    195                   548   
    3         30               4                    195                   991   
    4          2               3                     66                   462   
    
       user_location_city  orig_destination_distance  user_id  is_mobile  \
    0               48862                  2234.2641       12          0   
    1               52078                     0.0000      756          0   
    2               56440                     0.0000     1048          0   
    3               47725                     0.0000     1048          0   
    4               41898                  2454.8588     1482          0   
    
       is_package  channel  ...   78   80   81   82   83   85   90   91   95   99  
    0           1        9  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  
    1           1        4  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  
    2           1        9  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  
    3           0        9  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  
    4           1        1  ...  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0  
    
    [5 rows x 220 columns]
    

### Algorithms- separate the target variable and predicter variables.


```python
X = train_book.drop(['user_id', 'hotel_cluster', 'is_booking'], axis=1)
y = train_book.hotel_cluster
X.shape, y.shape
```




    ((64, 217), (64,))




```python
# Check if all of the 100 clusters are present in the training data.
y.nunique()
```




    44



### 1. Support Vector Machine (SVM)


```python
classifier = make_pipeline(preprocessing.StandardScaler(), svm.SVC(decision_function_shape='ovo'))
np.mean(cross_val_score(classifier, X, y, cv=4))
```




    0.0625



### 2. Naive Bayes classifier


```python
classifier = make_pipeline(preprocessing.StandardScaler(), GaussianNB(priors=None))
np.mean(cross_val_score(classifier, X, y, cv=4))
```




    0.3125



### 3. Logistic Regression


```python
classifier = make_pipeline(preprocessing.StandardScaler(), LogisticRegression(multi_class='ovr'))
np.mean(cross_val_score(classifier, X, y, cv=4))
```




    0.390625



### 4. K-Nearest Neighbor classifier


```python
classifier = make_pipeline(preprocessing.StandardScaler(), KNeighborsClassifier(n_neighbors=5))
np.mean(cross_val_score(classifier, X, y, cv=4, scoring='accuracy'))
```




    0.109375



SVM performed the best. Yet, the cross validation score is only 0.44. Other algorithms performed worse than that. Further feature engineering and increasing the number of folds might help improving the score.
The one pager summary for this approach is included in this notebook to keep the method coherent. 

### Summary

After completing the Exploratory Data Analysis (EDA), we got the idea to select the following algorithms based on the understanding of the datasets. In the following you will find the selected algorithms:

First, the  Support Vector Machine (SVM) performs classification by finding the hyperplane that maximizes the margin between the two classes. The vectors (cases) that define the hyperplane are the support vectors.  SVM can do both classification and regression. The clusters are multi-level (100) and used non-linear SVM.  Non-linear SVM means that the boundary that the algorithm calculates doesn't have to be a straight line. The advantage is that we can capture much more complex relationships between the data points without having to perform difficult transformations. The downside is that the training time is much longer as it's much more computationally intensive. Using SVM, help to achieve the highest cross-validation score.

Second, using the Naive Bayes classifier, which assumes that the presence (or absence) of a feature of a class is unrelated to the presence (or absence) of any other feature, given the class variable. Naive Bayes uses a similar method to predict the probability of different classes based on various attributes. This algorithm is mostly used in text classification and with problems having multiple classes. But it has the worst performance of the four models. Therefore, this classifier is not recommended for the problem at hand.

Third, logistic regression is the appropriate regression analysis to conduct when the dependent variable is dichotomous (binary). It is used to describe data and to explain the relationship between one dependent binary variable and one or more nominal, ordinal, interval, or ratio-level independent variables. The hotel falls in a specific cluster (yes/no) based on the chosen features. Logistic Regression was close to the performance of SVM but slightly worse.

Fourth, the K nearest neighbors is a simple algorithm that stores all available cases and classifies new cases based on a similarity measure (e.g., distance functions). It has been used in statistical estimation and pattern recognition already at the beginning of the 1970s as a non-parametric technique. The idea of KNN is to teach the model which users (with other similar characteristics) chose which hotel cluster and predict future cluster assignment based on that learning.
KNN works by finding the distances between a query and all the examples in the data, selecting the specified number examples (K) closest to the query, then votes for the most frequent label (in the case of classification) or averages the labels (in the case of regression).

KNN performed very similar to Logistic Regression for the model in question.



