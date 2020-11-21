
---
title: "Predictive-Analytics-Female Genital Mutilation"
date: 2020-11-20
tags: [data science, data wrangling, messy data]
header:
  image: "/images/perceptron/percept.jpg"
excerpt: "Data Science, Data Wrangling, Messy Data"
mathjax: "true"
---


DSC530-T301 Term Project
Female Genital Mutilation (FGM)
The report consists of female genital mutilation (FGM). I think this is an interesting topic with a large dataset under "Tableau Community Forums," but I am going to limit my analysis to some areas. The data of female genital mutilation (FGM) are under UNICEF(United Nations International Children's Fund) website.
Data Source: 
[https://community.tableau.com/docs/DOC-10635]
ANACONDA NAVIGATOR 
Jupyter Notebook 6.0.1
PYTHON 3.74
Required Packages
•	Numpy: for basic numerical computation 
•	SciPy: for scientific  computation including statistics
•	StatsModels: for regression and other statistical analysis
•	 Pandas: for representing and analyzing data  
•	Matplotlib: for visualization








#DSC630 Data Exploration

data <- read.csv("C:/Users/14026/Desktop/Data Science Courses 2020/DSC530_11252019/My project_DSC530/fusion_GLOBAL_DATAFLOW_UNICEF_1.0_.PT_F_15-49_FGM+PT_M_15-49_FGM_ELIM+PT_F_0-14_FGM+PT_F_15-49_FGM_ELIM.._(1).csv")
head(data)

dim(data)                   #shows the dimensions of the data frame by row and column
str(data)                   # shows the structure of the data frame
#summary(data)              # provides summary statistics on the columns of the data frame
colnames(data)              # shows the name of each column in the data frame
#head(data)                 #shows the first 6 rows of the data frame
#tail(data)                 #shows the last 6 rows of the data frame
#View(data)                 #shows a spreadsheet-like display of the entire data frame
#rownames(data)
#nrow(data)
#ncol(data)
#colnames(data)


library(dplyr)
glimpse(data)              # Explore the data


cor.test(data$Observation.Value, data$Time.Period)
#cor(data$Observation.Value, data$Time.Period, method = c("pearson", "kendall", "spearman"))
data.cor= cor(data$Observation.Value, data$Time.Period, method = c("pearson"))
data.cor

cov(data$Observation.Value, data$Time.Period)  #result [1] -3.865924

library(Hmisc)
data.rcorr = rcorr(as.matrix(data$Observation.Value, data$Time.Period))


#Visualizing the correlation matrix

install.packages("corrplot")
library(corrplot)
corrplot(data.rcorr)

palette = colorRampPalette(c("green", "white", "red")) (20)
heatmap(x = data.rcorr, col = palette, symm = TRUE)

# regression

#Run a regression analysis where Time.Period predicts Observation.Value.
mod <- lm(Time.Period ~ Observation.Value, data)
mod

# prediction

predict(mod, Time.Period=Observation.Value)

