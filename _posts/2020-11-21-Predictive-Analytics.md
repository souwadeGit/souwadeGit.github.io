
---
title: "Predictive-Analytics-Female Genital Mutilation (FGM)"
date: 2020-09-13
tags: [data science, data wrangling, messy data]
header:
  image: "/images/perceptron/percept.jpg"
excerpt: "Data Science, Data Wrangling, Messy Data"
mathjax: "true"
---

Predictive-Analytics

DSC530-T301 Term Project
Female Genital Mutilation (FGM)
The report consists of female genital mutilation (FGM). I think this is an interesting topic with a large dataset under "Tableau Community Forums," but I am going to limit my analysis to some areas. The data of female genital mutilation (FGM) are under UNICEF(United Nations International Children's Fund) website.
Data Source: 
https://community.tableau.com/docs/DOC-10635
ANACONDA NAVIGATOR 
Jupyter Notebook 6.0.1
PYTHON 3.74
Required Packages
•	Numpy: for basic numerical computation 
•	SciPy: for scientific  computation including statistics
•	StatsModels: for regression and other statistical analysis
•	 Pandas: for representing and analyzing data  
•	Matplotlib: for visualization




```python
import pandas as pd                     
import matplotlib.pyplot as plt          # plotting
import numpy as np                       # dense matrices
from scipy.sparse import csr_matrix      # sparse matrices
%matplotlib inline

import numpy as np

df = pd.read_csv("C:/Users/14026/Desktop/Data Science Courses 2020/DSC530_11252019/My project_DSC530/fusion_GLOBAL_DATAFLOW_UNICEF_1.0_.PT_F_15-49_FGM+PT_M_15-49_FGM_ELIM+PT_F_0-14_FGM+PT_F_15-49_FGM_ELIM.._(1).csv")
df.head(5)
#df.describe().head(5)
#df
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
      <th>Dataflow</th>
      <th>Geographic Area</th>
      <th>Indicator</th>
      <th>Sex</th>
      <th>Time Period</th>
      <th>Observation Value</th>
      <th>Unit Multiplier</th>
      <th>Unit of Measure</th>
      <th>Observation Status</th>
      <th>Observation Confidentaility</th>
      <th>...</th>
      <th>Weighted Sample Size</th>
      <th>Observation Footnote</th>
      <th>Series Footnote</th>
      <th>Data Source</th>
      <th>Citation of or link to the data source</th>
      <th>Custodian</th>
      <th>Time period activity related to when the data are collected</th>
      <th>Reference Period</th>
      <th>The period of time for which data are provided</th>
      <th>Current Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Cross-sector Indicators</td>
      <td>Côte d'Ivoire</td>
      <td>Percentage of girls (aged 0-14 years) who have...</td>
      <td>Total</td>
      <td>2016</td>
      <td>10.1</td>
      <td>NaN</td>
      <td>%</td>
      <td>Reanalysed</td>
      <td>Free</td>
      <td>...</td>
      <td>NaN</td>
      <td>Due to an error in the syntax used for the fin...</td>
      <td>NaN</td>
      <td>MICS 2016</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>End of fieldwork</td>
      <td>NaN</td>
      <td>2016</td>
      <td>Under 15 years old</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cross-sector Indicators</td>
      <td>Egypt</td>
      <td>Percentage of girls (aged 0-14 years) who have...</td>
      <td>Total</td>
      <td>2015</td>
      <td>14.1</td>
      <td>NaN</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>...</td>
      <td>NaN</td>
      <td>Age group is 1-14 years rather than 0-14 years</td>
      <td>NaN</td>
      <td>Health Issues Survey (DHS) 2015</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>End of fieldwork</td>
      <td>NaN</td>
      <td>2015</td>
      <td>Under 15 years old</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cross-sector Indicators</td>
      <td>Ethiopia</td>
      <td>Percentage of girls (aged 0-14 years) who have...</td>
      <td>Total</td>
      <td>2016</td>
      <td>15.7</td>
      <td>NaN</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>DHS 2016</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>End of fieldwork</td>
      <td>NaN</td>
      <td>2016</td>
      <td>Under 15 years old</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Cross-sector Indicators</td>
      <td>Guinea</td>
      <td>Percentage of girls (aged 0-14 years) who have...</td>
      <td>Total</td>
      <td>2016</td>
      <td>45.1</td>
      <td>NaN</td>
      <td>%</td>
      <td>Reanalysed</td>
      <td>Free</td>
      <td>...</td>
      <td>NaN</td>
      <td>Due to an error in the syntax used for the fin...</td>
      <td>NaN</td>
      <td>MICS 2016</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>End of fieldwork</td>
      <td>NaN</td>
      <td>2016</td>
      <td>Under 15 years old</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Cross-sector Indicators</td>
      <td>Iraq</td>
      <td>Percentage of girls (aged 0-14 years) who have...</td>
      <td>Total</td>
      <td>2018</td>
      <td>0.5</td>
      <td>NaN</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>MICS 2018</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>End of fieldwork</td>
      <td>NaN</td>
      <td>2018</td>
      <td>Under 15 years old</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>




```python

```


```python
#Get a list of column in the dataset
df.columns

```




    Index(['Dataflow', 'Geographic Area', 'Indicator', 'Sex', 'Time Period',
           'Observation Value', 'Unit Multiplier', 'Unit of Measure',
           'Observation Status', 'Observation Confidentaility', 'Lower Bound',
           'Upper Bound', 'Weighted Sample Size', 'Observation Footnote',
           'Series Footnote', 'Data Source',
           'Citation of or link to the data source', 'Custodian',
           'Time period activity related to when the data are collected',
           'Reference Period', 'The period of time for which data are provided',
           'Current Age'],
          dtype='object')




```python
df.dropna(axis='columns')
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
      <th>Dataflow</th>
      <th>Geographic Area</th>
      <th>Indicator</th>
      <th>Sex</th>
      <th>Time Period</th>
      <th>Observation Value</th>
      <th>Unit of Measure</th>
      <th>Observation Status</th>
      <th>Observation Confidentaility</th>
      <th>Data Source</th>
      <th>Time period activity related to when the data are collected</th>
      <th>The period of time for which data are provided</th>
      <th>Current Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Cross-sector Indicators</td>
      <td>Côte d'Ivoire</td>
      <td>Percentage of girls (aged 0-14 years) who have...</td>
      <td>Total</td>
      <td>2016</td>
      <td>10.1</td>
      <td>%</td>
      <td>Reanalysed</td>
      <td>Free</td>
      <td>MICS 2016</td>
      <td>End of fieldwork</td>
      <td>2016</td>
      <td>Under 15 years old</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cross-sector Indicators</td>
      <td>Egypt</td>
      <td>Percentage of girls (aged 0-14 years) who have...</td>
      <td>Total</td>
      <td>2015</td>
      <td>14.1</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>Health Issues Survey (DHS) 2015</td>
      <td>End of fieldwork</td>
      <td>2015</td>
      <td>Under 15 years old</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cross-sector Indicators</td>
      <td>Ethiopia</td>
      <td>Percentage of girls (aged 0-14 years) who have...</td>
      <td>Total</td>
      <td>2016</td>
      <td>15.7</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>DHS 2016</td>
      <td>End of fieldwork</td>
      <td>2016</td>
      <td>Under 15 years old</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Cross-sector Indicators</td>
      <td>Guinea</td>
      <td>Percentage of girls (aged 0-14 years) who have...</td>
      <td>Total</td>
      <td>2016</td>
      <td>45.1</td>
      <td>%</td>
      <td>Reanalysed</td>
      <td>Free</td>
      <td>MICS 2016</td>
      <td>End of fieldwork</td>
      <td>2016</td>
      <td>Under 15 years old</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Cross-sector Indicators</td>
      <td>Iraq</td>
      <td>Percentage of girls (aged 0-14 years) who have...</td>
      <td>Total</td>
      <td>2018</td>
      <td>0.5</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>MICS 2018</td>
      <td>End of fieldwork</td>
      <td>2018</td>
      <td>Under 15 years old</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Cross-sector Indicators</td>
      <td>Mali</td>
      <td>Percentage of girls (aged 0-14 years) who have...</td>
      <td>Total</td>
      <td>2015</td>
      <td>73.2</td>
      <td>%</td>
      <td>Reanalysed</td>
      <td>Free</td>
      <td>MICS 2015</td>
      <td>End of fieldwork</td>
      <td>2015</td>
      <td>Under 15 years old</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Cross-sector Indicators</td>
      <td>Mauritania</td>
      <td>Percentage of girls (aged 0-14 years) who have...</td>
      <td>Total</td>
      <td>2015</td>
      <td>51.4</td>
      <td>%</td>
      <td>Reanalysed</td>
      <td>Free</td>
      <td>MICS 2015</td>
      <td>End of fieldwork</td>
      <td>2015</td>
      <td>Under 15 years old</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Cross-sector Indicators</td>
      <td>Nigeria</td>
      <td>Percentage of girls (aged 0-14 years) who have...</td>
      <td>Total</td>
      <td>2017</td>
      <td>12.7</td>
      <td>%</td>
      <td>Reanalysed</td>
      <td>Free</td>
      <td>MICS 2016-17</td>
      <td>End of fieldwork</td>
      <td>2016-17</td>
      <td>Under 15 years old</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Cross-sector Indicators</td>
      <td>Senegal</td>
      <td>Percentage of girls (aged 0-14 years) who have...</td>
      <td>Total</td>
      <td>2017</td>
      <td>13.9</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>DHS continuous 2017</td>
      <td>End of fieldwork</td>
      <td>2017</td>
      <td>Under 15 years old</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Cross-sector Indicators</td>
      <td>Sierra Leone</td>
      <td>Percentage of girls (aged 0-14 years) who have...</td>
      <td>Total</td>
      <td>2017</td>
      <td>8.4</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>MICS 2017</td>
      <td>End of fieldwork</td>
      <td>2017</td>
      <td>Under 15 years old</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Cross-sector Indicators</td>
      <td>United Republic of Tanzania</td>
      <td>Percentage of girls (aged 0-14 years) who have...</td>
      <td>Total</td>
      <td>2016</td>
      <td>0.4</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>DHS 2015-16</td>
      <td>End of fieldwork</td>
      <td>2015-16</td>
      <td>Under 15 years old</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Cross-sector Indicators</td>
      <td>Chad</td>
      <td>Percentage of girls and women (aged 15-49 year...</td>
      <td>Total</td>
      <td>2015</td>
      <td>38.4</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>DHS 2014-15</td>
      <td>End of fieldwork</td>
      <td>2014-15</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Cross-sector Indicators</td>
      <td>Côte d'Ivoire</td>
      <td>Percentage of girls and women (aged 15-49 year...</td>
      <td>Total</td>
      <td>2016</td>
      <td>36.7</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>MICS 2016</td>
      <td>End of fieldwork</td>
      <td>2016</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Cross-sector Indicators</td>
      <td>Egypt</td>
      <td>Percentage of girls and women (aged 15-49 year...</td>
      <td>Total</td>
      <td>2015</td>
      <td>87.2</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>Health Issues Survey (DHS) 2015</td>
      <td>End of fieldwork</td>
      <td>2015</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Cross-sector Indicators</td>
      <td>Ethiopia</td>
      <td>Percentage of girls and women (aged 15-49 year...</td>
      <td>Total</td>
      <td>2016</td>
      <td>65.2</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>DHS 2016</td>
      <td>End of fieldwork</td>
      <td>2016</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Cross-sector Indicators</td>
      <td>Guinea</td>
      <td>Percentage of girls and women (aged 15-49 year...</td>
      <td>Total</td>
      <td>2016</td>
      <td>96.8</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>MICS 2016</td>
      <td>End of fieldwork</td>
      <td>2016</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Cross-sector Indicators</td>
      <td>Iraq</td>
      <td>Percentage of girls and women (aged 15-49 year...</td>
      <td>Total</td>
      <td>2018</td>
      <td>7.4</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>MICS 2018</td>
      <td>End of fieldwork</td>
      <td>2018</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Cross-sector Indicators</td>
      <td>Mali</td>
      <td>Percentage of girls and women (aged 15-49 year...</td>
      <td>Total</td>
      <td>2015</td>
      <td>82.7</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>MICS 2015</td>
      <td>End of fieldwork</td>
      <td>2015</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Cross-sector Indicators</td>
      <td>Mauritania</td>
      <td>Percentage of girls and women (aged 15-49 year...</td>
      <td>Total</td>
      <td>2015</td>
      <td>66.6</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>MICS 2015</td>
      <td>End of fieldwork</td>
      <td>2015</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Cross-sector Indicators</td>
      <td>Nigeria</td>
      <td>Percentage of girls and women (aged 15-49 year...</td>
      <td>Total</td>
      <td>2017</td>
      <td>18.4</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>MICS 2016-17</td>
      <td>End of fieldwork</td>
      <td>2016-17</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Cross-sector Indicators</td>
      <td>Senegal</td>
      <td>Percentage of girls and women (aged 15-49 year...</td>
      <td>Total</td>
      <td>2017</td>
      <td>24.0</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>DHS Continuous 2017</td>
      <td>End of fieldwork</td>
      <td>2017</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Cross-sector Indicators</td>
      <td>Sierra Leone</td>
      <td>Percentage of girls and women (aged 15-49 year...</td>
      <td>Total</td>
      <td>2017</td>
      <td>86.1</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>MICS 2017</td>
      <td>End of fieldwork</td>
      <td>2017</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Cross-sector Indicators</td>
      <td>Uganda</td>
      <td>Percentage of girls and women (aged 15-49 year...</td>
      <td>Total</td>
      <td>2016</td>
      <td>0.3</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>DHS 2016</td>
      <td>End of fieldwork</td>
      <td>2016</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Cross-sector Indicators</td>
      <td>United Republic of Tanzania</td>
      <td>Percentage of girls and women (aged 15-49 year...</td>
      <td>Total</td>
      <td>2016</td>
      <td>10.0</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>DHS 2015-16</td>
      <td>End of fieldwork</td>
      <td>2015-16</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Cross-sector Indicators</td>
      <td>Chad</td>
      <td>Percentage of women (aged 15-49 years) who thi...</td>
      <td>Total</td>
      <td>2015</td>
      <td>45.1</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>DHS 2014-15</td>
      <td>End of fieldwork</td>
      <td>2014-15</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Cross-sector Indicators</td>
      <td>Egypt</td>
      <td>Percentage of women (aged 15-49 years) who thi...</td>
      <td>Total</td>
      <td>2015</td>
      <td>37.5</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>Health Issues Survey (DHS) 2015</td>
      <td>End of fieldwork</td>
      <td>2015</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Cross-sector Indicators</td>
      <td>United Republic of Tanzania</td>
      <td>Percentage of women (aged 15-49 years) who thi...</td>
      <td>Total</td>
      <td>2016</td>
      <td>95.0</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>DHS 2015-16</td>
      <td>End of fieldwork</td>
      <td>2015-16</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Cross-sector Indicators</td>
      <td>Ethiopia</td>
      <td>Percentage of women (aged 15-49 years) who thi...</td>
      <td>Total</td>
      <td>2016</td>
      <td>79.3</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>DHS 2016</td>
      <td>End of fieldwork</td>
      <td>2016</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Cross-sector Indicators</td>
      <td>Mauritania</td>
      <td>Percentage of women (aged 15-49 years) who thi...</td>
      <td>Total</td>
      <td>2015</td>
      <td>49.5</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>MICS  2015</td>
      <td>End of fieldwork</td>
      <td>2015</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Cross-sector Indicators</td>
      <td>Guinea</td>
      <td>Percentage of women (aged 15-49 years) who thi...</td>
      <td>Total</td>
      <td>2016</td>
      <td>21.5</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>MICS  2016</td>
      <td>End of fieldwork</td>
      <td>2016</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Cross-sector Indicators</td>
      <td>Côte d'Ivoire</td>
      <td>Percentage of women (aged 15-49 years) who thi...</td>
      <td>Total</td>
      <td>2016</td>
      <td>79.4</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>MICS 2016</td>
      <td>End of fieldwork</td>
      <td>2016</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Cross-sector Indicators</td>
      <td>Nigeria</td>
      <td>Percentage of women (aged 15-49 years) who thi...</td>
      <td>Total</td>
      <td>2017</td>
      <td>67.5</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>MICS 2016-17</td>
      <td>End of fieldwork</td>
      <td>2016-17</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Cross-sector Indicators</td>
      <td>Mali</td>
      <td>Percentage of women (aged 15-49 years) who thi...</td>
      <td>Total</td>
      <td>2015</td>
      <td>14.4</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>MICS  2015</td>
      <td>End of fieldwork</td>
      <td>2015</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Cross-sector Indicators</td>
      <td>Sierra Leone</td>
      <td>Percentage of women (aged 15-49 years) who thi...</td>
      <td>Total</td>
      <td>2017</td>
      <td>26.8</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>MICS 2017</td>
      <td>End of fieldwork</td>
      <td>2017</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Cross-sector Indicators</td>
      <td>Iraq</td>
      <td>Percentage of women (aged 15-49 years) who thi...</td>
      <td>Total</td>
      <td>2018</td>
      <td>93.6</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>MICS 2018</td>
      <td>End of fieldwork</td>
      <td>2018</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Cross-sector Indicators</td>
      <td>Senegal</td>
      <td>Percentage of women (aged 15-49 years) who thi...</td>
      <td>Total</td>
      <td>2017</td>
      <td>80.9</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>DHS 2017</td>
      <td>End of fieldwork</td>
      <td>2017</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Cross-sector Indicators</td>
      <td>Ethiopia</td>
      <td>Percentage of boys and men (aged 15-49 years) ...</td>
      <td>Total</td>
      <td>2016</td>
      <td>86.7</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>DHS 2016</td>
      <td>End of fieldwork</td>
      <td>2016</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Cross-sector Indicators</td>
      <td>Senegal</td>
      <td>Percentage of boys and men (aged 15-49 years) ...</td>
      <td>Total</td>
      <td>2017</td>
      <td>78.8</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>DHS 2017</td>
      <td>End of fieldwork</td>
      <td>2017</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Cross-sector Indicators</td>
      <td>Egypt</td>
      <td>Percentage of boys and men (aged 15-49 years) ...</td>
      <td>Total</td>
      <td>2015</td>
      <td>27.9</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>HIS 2015</td>
      <td>End of fieldwork</td>
      <td>2015</td>
      <td>15 to 49 years old</td>
    </tr>
  </tbody>
</table>
</div>




```python
# How to identify the null value NaN where the value is not equal to 0

#df.isnull().head()
df.isnull().sum()

#df[df['Unit Multiplier'].isnull()]


```




    Dataflow                                                        0
    Geographic Area                                                 0
    Indicator                                                       0
    Sex                                                             0
    Time Period                                                     0
    Observation Value                                               0
    Unit Multiplier                                                39
    Unit of Measure                                                 0
    Observation Status                                              0
    Observation Confidentaility                                     0
    Lower Bound                                                    39
    Upper Bound                                                    39
    Weighted Sample Size                                           39
    Observation Footnote                                           33
    Series Footnote                                                39
    Data Source                                                     0
    Citation of or link to the data source                         39
    Custodian                                                      39
    Time period activity related to when the data are collected     0
    Reference Period                                               39
    The period of time for which data are provided                  0
    Current Age                                                     0
    dtype: int64




```python
df.columns
#df
```




    Index(['Dataflow', 'Geographic Area', 'Indicator', 'Sex', 'Time Period',
           'Observation Value', 'Unit Multiplier', 'Unit of Measure',
           'Observation Status', 'Observation Confidentaility', 'Lower Bound',
           'Upper Bound', 'Weighted Sample Size', 'Observation Footnote',
           'Series Footnote', 'Data Source',
           'Citation of or link to the data source', 'Custodian',
           'Time period activity related to when the data are collected',
           'Reference Period', 'The period of time for which data are provided',
           'Current Age'],
          dtype='object')




```python
# How to identify the null value NaN where the value is equal to 0

#df.notnull().head()
df.notnull().sum()
```




    Dataflow                                                       39
    Geographic Area                                                39
    Indicator                                                      39
    Sex                                                            39
    Time Period                                                    39
    Observation Value                                              39
    Unit Multiplier                                                 0
    Unit of Measure                                                39
    Observation Status                                             39
    Observation Confidentaility                                    39
    Lower Bound                                                     0
    Upper Bound                                                     0
    Weighted Sample Size                                            0
    Observation Footnote                                            6
    Series Footnote                                                 0
    Data Source                                                    39
    Citation of or link to the data source                          0
    Custodian                                                       0
    Time period activity related to when the data are collected    39
    Reference Period                                                0
    The period of time for which data are provided                 39
    Current Age                                                    39
    dtype: int64




```python
df.dropna(how ='all').shape
```




    (39, 22)




```python
df.dropna( how ='all').tail()
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
      <th>Dataflow</th>
      <th>Geographic Area</th>
      <th>Indicator</th>
      <th>Sex</th>
      <th>Time Period</th>
      <th>Observation Value</th>
      <th>Unit Multiplier</th>
      <th>Unit of Measure</th>
      <th>Observation Status</th>
      <th>Observation Confidentaility</th>
      <th>...</th>
      <th>Weighted Sample Size</th>
      <th>Observation Footnote</th>
      <th>Series Footnote</th>
      <th>Data Source</th>
      <th>Citation of or link to the data source</th>
      <th>Custodian</th>
      <th>Time period activity related to when the data are collected</th>
      <th>Reference Period</th>
      <th>The period of time for which data are provided</th>
      <th>Current Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>34</th>
      <td>Cross-sector Indicators</td>
      <td>Iraq</td>
      <td>Percentage of women (aged 15-49 years) who thi...</td>
      <td>Total</td>
      <td>2018</td>
      <td>93.6</td>
      <td>NaN</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>MICS 2018</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>End of fieldwork</td>
      <td>NaN</td>
      <td>2018</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Cross-sector Indicators</td>
      <td>Senegal</td>
      <td>Percentage of women (aged 15-49 years) who thi...</td>
      <td>Total</td>
      <td>2017</td>
      <td>80.9</td>
      <td>NaN</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>DHS 2017</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>End of fieldwork</td>
      <td>NaN</td>
      <td>2017</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Cross-sector Indicators</td>
      <td>Ethiopia</td>
      <td>Percentage of boys and men (aged 15-49 years) ...</td>
      <td>Total</td>
      <td>2016</td>
      <td>86.7</td>
      <td>NaN</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>DHS 2016</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>End of fieldwork</td>
      <td>NaN</td>
      <td>2016</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Cross-sector Indicators</td>
      <td>Senegal</td>
      <td>Percentage of boys and men (aged 15-49 years) ...</td>
      <td>Total</td>
      <td>2017</td>
      <td>78.8</td>
      <td>NaN</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>DHS 2017</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>End of fieldwork</td>
      <td>NaN</td>
      <td>2017</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Cross-sector Indicators</td>
      <td>Egypt</td>
      <td>Percentage of boys and men (aged 15-49 years) ...</td>
      <td>Total</td>
      <td>2015</td>
      <td>27.9</td>
      <td>NaN</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>HIS 2015</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>End of fieldwork</td>
      <td>NaN</td>
      <td>2015</td>
      <td>15 to 49 years old</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>




```python
clean_df = df.dropna(axis='columns')
clean_df.head(5) 
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
      <th>Dataflow</th>
      <th>Geographic Area</th>
      <th>Indicator</th>
      <th>Sex</th>
      <th>Time Period</th>
      <th>Observation Value</th>
      <th>Unit of Measure</th>
      <th>Observation Status</th>
      <th>Observation Confidentaility</th>
      <th>Data Source</th>
      <th>Time period activity related to when the data are collected</th>
      <th>The period of time for which data are provided</th>
      <th>Current Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Cross-sector Indicators</td>
      <td>Côte d'Ivoire</td>
      <td>Percentage of girls (aged 0-14 years) who have...</td>
      <td>Total</td>
      <td>2016</td>
      <td>10.1</td>
      <td>%</td>
      <td>Reanalysed</td>
      <td>Free</td>
      <td>MICS 2016</td>
      <td>End of fieldwork</td>
      <td>2016</td>
      <td>Under 15 years old</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cross-sector Indicators</td>
      <td>Egypt</td>
      <td>Percentage of girls (aged 0-14 years) who have...</td>
      <td>Total</td>
      <td>2015</td>
      <td>14.1</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>Health Issues Survey (DHS) 2015</td>
      <td>End of fieldwork</td>
      <td>2015</td>
      <td>Under 15 years old</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cross-sector Indicators</td>
      <td>Ethiopia</td>
      <td>Percentage of girls (aged 0-14 years) who have...</td>
      <td>Total</td>
      <td>2016</td>
      <td>15.7</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>DHS 2016</td>
      <td>End of fieldwork</td>
      <td>2016</td>
      <td>Under 15 years old</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Cross-sector Indicators</td>
      <td>Guinea</td>
      <td>Percentage of girls (aged 0-14 years) who have...</td>
      <td>Total</td>
      <td>2016</td>
      <td>45.1</td>
      <td>%</td>
      <td>Reanalysed</td>
      <td>Free</td>
      <td>MICS 2016</td>
      <td>End of fieldwork</td>
      <td>2016</td>
      <td>Under 15 years old</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Cross-sector Indicators</td>
      <td>Iraq</td>
      <td>Percentage of girls (aged 0-14 years) who have...</td>
      <td>Total</td>
      <td>2018</td>
      <td>0.5</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>MICS 2018</td>
      <td>End of fieldwork</td>
      <td>2018</td>
      <td>Under 15 years old</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.dropna(subset = ['Unit Multiplier'],how='all').shape
```




    (0, 22)




```python
df.dropna(subset = ['Observation Footnote'],how='any').shape
```




    (6, 22)




```python
df.dropna(subset = ['Time period activity related to when the data are collected'],how='all').shape
```




    (39, 22)




```python
#Python Pandas Tutorial 16 | How to Fill Up NA Values | Various ways to fill missing values in python
df.fillna(method='ffill')
df.fillna(0)
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
      <th>Dataflow</th>
      <th>Geographic Area</th>
      <th>Indicator</th>
      <th>Sex</th>
      <th>Time Period</th>
      <th>Observation Value</th>
      <th>Unit Multiplier</th>
      <th>Unit of Measure</th>
      <th>Observation Status</th>
      <th>Observation Confidentaility</th>
      <th>...</th>
      <th>Weighted Sample Size</th>
      <th>Observation Footnote</th>
      <th>Series Footnote</th>
      <th>Data Source</th>
      <th>Citation of or link to the data source</th>
      <th>Custodian</th>
      <th>Time period activity related to when the data are collected</th>
      <th>Reference Period</th>
      <th>The period of time for which data are provided</th>
      <th>Current Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Cross-sector Indicators</td>
      <td>Côte d'Ivoire</td>
      <td>Percentage of girls (aged 0-14 years) who have...</td>
      <td>Total</td>
      <td>2016</td>
      <td>10.1</td>
      <td>0.0</td>
      <td>%</td>
      <td>Reanalysed</td>
      <td>Free</td>
      <td>...</td>
      <td>0.0</td>
      <td>Due to an error in the syntax used for the fin...</td>
      <td>0.0</td>
      <td>MICS 2016</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>End of fieldwork</td>
      <td>0.0</td>
      <td>2016</td>
      <td>Under 15 years old</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cross-sector Indicators</td>
      <td>Egypt</td>
      <td>Percentage of girls (aged 0-14 years) who have...</td>
      <td>Total</td>
      <td>2015</td>
      <td>14.1</td>
      <td>0.0</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>...</td>
      <td>0.0</td>
      <td>Age group is 1-14 years rather than 0-14 years</td>
      <td>0.0</td>
      <td>Health Issues Survey (DHS) 2015</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>End of fieldwork</td>
      <td>0.0</td>
      <td>2015</td>
      <td>Under 15 years old</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cross-sector Indicators</td>
      <td>Ethiopia</td>
      <td>Percentage of girls (aged 0-14 years) who have...</td>
      <td>Total</td>
      <td>2016</td>
      <td>15.7</td>
      <td>0.0</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>DHS 2016</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>End of fieldwork</td>
      <td>0.0</td>
      <td>2016</td>
      <td>Under 15 years old</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Cross-sector Indicators</td>
      <td>Guinea</td>
      <td>Percentage of girls (aged 0-14 years) who have...</td>
      <td>Total</td>
      <td>2016</td>
      <td>45.1</td>
      <td>0.0</td>
      <td>%</td>
      <td>Reanalysed</td>
      <td>Free</td>
      <td>...</td>
      <td>0.0</td>
      <td>Due to an error in the syntax used for the fin...</td>
      <td>0.0</td>
      <td>MICS 2016</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>End of fieldwork</td>
      <td>0.0</td>
      <td>2016</td>
      <td>Under 15 years old</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Cross-sector Indicators</td>
      <td>Iraq</td>
      <td>Percentage of girls (aged 0-14 years) who have...</td>
      <td>Total</td>
      <td>2018</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>MICS 2018</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>End of fieldwork</td>
      <td>0.0</td>
      <td>2018</td>
      <td>Under 15 years old</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Cross-sector Indicators</td>
      <td>Mali</td>
      <td>Percentage of girls (aged 0-14 years) who have...</td>
      <td>Total</td>
      <td>2015</td>
      <td>73.2</td>
      <td>0.0</td>
      <td>%</td>
      <td>Reanalysed</td>
      <td>Free</td>
      <td>...</td>
      <td>0.0</td>
      <td>Due to an error in the syntax used for the fin...</td>
      <td>0.0</td>
      <td>MICS 2015</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>End of fieldwork</td>
      <td>0.0</td>
      <td>2015</td>
      <td>Under 15 years old</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Cross-sector Indicators</td>
      <td>Mauritania</td>
      <td>Percentage of girls (aged 0-14 years) who have...</td>
      <td>Total</td>
      <td>2015</td>
      <td>51.4</td>
      <td>0.0</td>
      <td>%</td>
      <td>Reanalysed</td>
      <td>Free</td>
      <td>...</td>
      <td>0.0</td>
      <td>Due to an error in the syntax used for the fin...</td>
      <td>0.0</td>
      <td>MICS 2015</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>End of fieldwork</td>
      <td>0.0</td>
      <td>2015</td>
      <td>Under 15 years old</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Cross-sector Indicators</td>
      <td>Nigeria</td>
      <td>Percentage of girls (aged 0-14 years) who have...</td>
      <td>Total</td>
      <td>2017</td>
      <td>12.7</td>
      <td>0.0</td>
      <td>%</td>
      <td>Reanalysed</td>
      <td>Free</td>
      <td>...</td>
      <td>0.0</td>
      <td>Due to an error in the syntax used for the fin...</td>
      <td>0.0</td>
      <td>MICS 2016-17</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>End of fieldwork</td>
      <td>0.0</td>
      <td>2016-17</td>
      <td>Under 15 years old</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Cross-sector Indicators</td>
      <td>Senegal</td>
      <td>Percentage of girls (aged 0-14 years) who have...</td>
      <td>Total</td>
      <td>2017</td>
      <td>13.9</td>
      <td>0.0</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>DHS continuous 2017</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>End of fieldwork</td>
      <td>0.0</td>
      <td>2017</td>
      <td>Under 15 years old</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Cross-sector Indicators</td>
      <td>Sierra Leone</td>
      <td>Percentage of girls (aged 0-14 years) who have...</td>
      <td>Total</td>
      <td>2017</td>
      <td>8.4</td>
      <td>0.0</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>MICS 2017</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>End of fieldwork</td>
      <td>0.0</td>
      <td>2017</td>
      <td>Under 15 years old</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Cross-sector Indicators</td>
      <td>United Republic of Tanzania</td>
      <td>Percentage of girls (aged 0-14 years) who have...</td>
      <td>Total</td>
      <td>2016</td>
      <td>0.4</td>
      <td>0.0</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>DHS 2015-16</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>End of fieldwork</td>
      <td>0.0</td>
      <td>2015-16</td>
      <td>Under 15 years old</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Cross-sector Indicators</td>
      <td>Chad</td>
      <td>Percentage of girls and women (aged 15-49 year...</td>
      <td>Total</td>
      <td>2015</td>
      <td>38.4</td>
      <td>0.0</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>DHS 2014-15</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>End of fieldwork</td>
      <td>0.0</td>
      <td>2014-15</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Cross-sector Indicators</td>
      <td>Côte d'Ivoire</td>
      <td>Percentage of girls and women (aged 15-49 year...</td>
      <td>Total</td>
      <td>2016</td>
      <td>36.7</td>
      <td>0.0</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>MICS 2016</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>End of fieldwork</td>
      <td>0.0</td>
      <td>2016</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Cross-sector Indicators</td>
      <td>Egypt</td>
      <td>Percentage of girls and women (aged 15-49 year...</td>
      <td>Total</td>
      <td>2015</td>
      <td>87.2</td>
      <td>0.0</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>Health Issues Survey (DHS) 2015</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>End of fieldwork</td>
      <td>0.0</td>
      <td>2015</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Cross-sector Indicators</td>
      <td>Ethiopia</td>
      <td>Percentage of girls and women (aged 15-49 year...</td>
      <td>Total</td>
      <td>2016</td>
      <td>65.2</td>
      <td>0.0</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>DHS 2016</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>End of fieldwork</td>
      <td>0.0</td>
      <td>2016</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Cross-sector Indicators</td>
      <td>Guinea</td>
      <td>Percentage of girls and women (aged 15-49 year...</td>
      <td>Total</td>
      <td>2016</td>
      <td>96.8</td>
      <td>0.0</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>MICS 2016</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>End of fieldwork</td>
      <td>0.0</td>
      <td>2016</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Cross-sector Indicators</td>
      <td>Iraq</td>
      <td>Percentage of girls and women (aged 15-49 year...</td>
      <td>Total</td>
      <td>2018</td>
      <td>7.4</td>
      <td>0.0</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>MICS 2018</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>End of fieldwork</td>
      <td>0.0</td>
      <td>2018</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Cross-sector Indicators</td>
      <td>Mali</td>
      <td>Percentage of girls and women (aged 15-49 year...</td>
      <td>Total</td>
      <td>2015</td>
      <td>82.7</td>
      <td>0.0</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>MICS 2015</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>End of fieldwork</td>
      <td>0.0</td>
      <td>2015</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Cross-sector Indicators</td>
      <td>Mauritania</td>
      <td>Percentage of girls and women (aged 15-49 year...</td>
      <td>Total</td>
      <td>2015</td>
      <td>66.6</td>
      <td>0.0</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>MICS 2015</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>End of fieldwork</td>
      <td>0.0</td>
      <td>2015</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Cross-sector Indicators</td>
      <td>Nigeria</td>
      <td>Percentage of girls and women (aged 15-49 year...</td>
      <td>Total</td>
      <td>2017</td>
      <td>18.4</td>
      <td>0.0</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>MICS 2016-17</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>End of fieldwork</td>
      <td>0.0</td>
      <td>2016-17</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Cross-sector Indicators</td>
      <td>Senegal</td>
      <td>Percentage of girls and women (aged 15-49 year...</td>
      <td>Total</td>
      <td>2017</td>
      <td>24.0</td>
      <td>0.0</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>DHS Continuous 2017</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>End of fieldwork</td>
      <td>0.0</td>
      <td>2017</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Cross-sector Indicators</td>
      <td>Sierra Leone</td>
      <td>Percentage of girls and women (aged 15-49 year...</td>
      <td>Total</td>
      <td>2017</td>
      <td>86.1</td>
      <td>0.0</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>MICS 2017</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>End of fieldwork</td>
      <td>0.0</td>
      <td>2017</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Cross-sector Indicators</td>
      <td>Uganda</td>
      <td>Percentage of girls and women (aged 15-49 year...</td>
      <td>Total</td>
      <td>2016</td>
      <td>0.3</td>
      <td>0.0</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>DHS 2016</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>End of fieldwork</td>
      <td>0.0</td>
      <td>2016</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Cross-sector Indicators</td>
      <td>United Republic of Tanzania</td>
      <td>Percentage of girls and women (aged 15-49 year...</td>
      <td>Total</td>
      <td>2016</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>DHS 2015-16</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>End of fieldwork</td>
      <td>0.0</td>
      <td>2015-16</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Cross-sector Indicators</td>
      <td>Chad</td>
      <td>Percentage of women (aged 15-49 years) who thi...</td>
      <td>Total</td>
      <td>2015</td>
      <td>45.1</td>
      <td>0.0</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>DHS 2014-15</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>End of fieldwork</td>
      <td>0.0</td>
      <td>2014-15</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Cross-sector Indicators</td>
      <td>Egypt</td>
      <td>Percentage of women (aged 15-49 years) who thi...</td>
      <td>Total</td>
      <td>2015</td>
      <td>37.5</td>
      <td>0.0</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>Health Issues Survey (DHS) 2015</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>End of fieldwork</td>
      <td>0.0</td>
      <td>2015</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Cross-sector Indicators</td>
      <td>United Republic of Tanzania</td>
      <td>Percentage of women (aged 15-49 years) who thi...</td>
      <td>Total</td>
      <td>2016</td>
      <td>95.0</td>
      <td>0.0</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>DHS 2015-16</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>End of fieldwork</td>
      <td>0.0</td>
      <td>2015-16</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Cross-sector Indicators</td>
      <td>Ethiopia</td>
      <td>Percentage of women (aged 15-49 years) who thi...</td>
      <td>Total</td>
      <td>2016</td>
      <td>79.3</td>
      <td>0.0</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>DHS 2016</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>End of fieldwork</td>
      <td>0.0</td>
      <td>2016</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Cross-sector Indicators</td>
      <td>Mauritania</td>
      <td>Percentage of women (aged 15-49 years) who thi...</td>
      <td>Total</td>
      <td>2015</td>
      <td>49.5</td>
      <td>0.0</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>MICS  2015</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>End of fieldwork</td>
      <td>0.0</td>
      <td>2015</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Cross-sector Indicators</td>
      <td>Guinea</td>
      <td>Percentage of women (aged 15-49 years) who thi...</td>
      <td>Total</td>
      <td>2016</td>
      <td>21.5</td>
      <td>0.0</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>MICS  2016</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>End of fieldwork</td>
      <td>0.0</td>
      <td>2016</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Cross-sector Indicators</td>
      <td>Côte d'Ivoire</td>
      <td>Percentage of women (aged 15-49 years) who thi...</td>
      <td>Total</td>
      <td>2016</td>
      <td>79.4</td>
      <td>0.0</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>MICS 2016</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>End of fieldwork</td>
      <td>0.0</td>
      <td>2016</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Cross-sector Indicators</td>
      <td>Nigeria</td>
      <td>Percentage of women (aged 15-49 years) who thi...</td>
      <td>Total</td>
      <td>2017</td>
      <td>67.5</td>
      <td>0.0</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>MICS 2016-17</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>End of fieldwork</td>
      <td>0.0</td>
      <td>2016-17</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Cross-sector Indicators</td>
      <td>Mali</td>
      <td>Percentage of women (aged 15-49 years) who thi...</td>
      <td>Total</td>
      <td>2015</td>
      <td>14.4</td>
      <td>0.0</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>MICS  2015</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>End of fieldwork</td>
      <td>0.0</td>
      <td>2015</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Cross-sector Indicators</td>
      <td>Sierra Leone</td>
      <td>Percentage of women (aged 15-49 years) who thi...</td>
      <td>Total</td>
      <td>2017</td>
      <td>26.8</td>
      <td>0.0</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>MICS 2017</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>End of fieldwork</td>
      <td>0.0</td>
      <td>2017</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Cross-sector Indicators</td>
      <td>Iraq</td>
      <td>Percentage of women (aged 15-49 years) who thi...</td>
      <td>Total</td>
      <td>2018</td>
      <td>93.6</td>
      <td>0.0</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>MICS 2018</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>End of fieldwork</td>
      <td>0.0</td>
      <td>2018</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Cross-sector Indicators</td>
      <td>Senegal</td>
      <td>Percentage of women (aged 15-49 years) who thi...</td>
      <td>Total</td>
      <td>2017</td>
      <td>80.9</td>
      <td>0.0</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>DHS 2017</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>End of fieldwork</td>
      <td>0.0</td>
      <td>2017</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Cross-sector Indicators</td>
      <td>Ethiopia</td>
      <td>Percentage of boys and men (aged 15-49 years) ...</td>
      <td>Total</td>
      <td>2016</td>
      <td>86.7</td>
      <td>0.0</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>DHS 2016</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>End of fieldwork</td>
      <td>0.0</td>
      <td>2016</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Cross-sector Indicators</td>
      <td>Senegal</td>
      <td>Percentage of boys and men (aged 15-49 years) ...</td>
      <td>Total</td>
      <td>2017</td>
      <td>78.8</td>
      <td>0.0</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>DHS 2017</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>End of fieldwork</td>
      <td>0.0</td>
      <td>2017</td>
      <td>15 to 49 years old</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Cross-sector Indicators</td>
      <td>Egypt</td>
      <td>Percentage of boys and men (aged 15-49 years) ...</td>
      <td>Total</td>
      <td>2015</td>
      <td>27.9</td>
      <td>0.0</td>
      <td>%</td>
      <td>Reported</td>
      <td>Free</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>HIS 2015</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>End of fieldwork</td>
      <td>0.0</td>
      <td>2015</td>
      <td>15 to 49 years old</td>
    </tr>
  </tbody>
</table>
<p>39 rows × 22 columns</p>
</div>




```python
# Data Exploration

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
      <th>Time Period</th>
      <th>Observation Value</th>
      <th>Unit Multiplier</th>
      <th>Lower Bound</th>
      <th>Upper Bound</th>
      <th>Weighted Sample Size</th>
      <th>Series Footnote</th>
      <th>Citation of or link to the data source</th>
      <th>Custodian</th>
      <th>Reference Period</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Time Period</th>
      <td>1.000000</td>
      <td>-0.127343</td>
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
      <th>Observation Value</th>
      <td>-0.127343</td>
      <td>1.000000</td>
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
      <th>Unit Multiplier</th>
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
      <th>Lower Bound</th>
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
      <th>Upper Bound</th>
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
      <th>Weighted Sample Size</th>
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
      <th>Series Footnote</th>
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
      <th>Citation of or link to the data source</th>
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
      <th>Custodian</th>
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
      <th>Reference Period</th>
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
</div>




```python
#df.hist()
```


```python
import matplotlib.pyplot as plt
#%matplotlib inline
# x= Time.Period = [2016 2015 2016 2016 2018 2015 2015 2017 2017 2017] 
# y= Observation.Value=[10.1 14.1 15.7 45.1 0.5 73.2 51.4 12.7 13.9 8.4

df.hist('Observation Value')
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x000002CD23AEFB20>]],
          dtype=object)




![png](output_16_1.png)



```python
df.hist('Time Period')
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x000002CD23B86490>]],
          dtype=object)




![png](output_17_1.png)


End
