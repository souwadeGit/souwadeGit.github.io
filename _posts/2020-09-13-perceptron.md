---
title: "DATA SCIENCE PROJEECT"
date: 2020-09-13
tags: [data science, data wrangling, messy data]
header:
  image: "/images/perceptron/percept.jpg"
excerpt: "Data Science, Data Wrangling, Messy Data"
mathjax: "true"
---

# Project 1: Proposal and Data Selection
## DSC680 Applied Data Science						09/013/2020
### Project 1: Proposal and Data Selection

**Which Domain?**
Data Sources: World Happiness Report
The dataset is under the Kaggle website. The World Happiness Report is a landmark survey of the state of global happiness. There are four reports published: the first in 2012, the second in 2013, the third in 2015, and the fourth in 2016. The dataset (World Happiness 2017), which ranks 155 countries by their happiness levels. It was released at the United Nations at an event celebrating the International Day of Happiness on March 20th. 
**Which Data?
world-happiness/2017.csv
world-happiness/2019.csv
world-happiness/2016.csv
world-happiness/2015.csv
world-happiness/2018.csv**
[link](https://www.kaggle.com/pinarkaya/world-happiness-eda-visualization-ml/notebook)?
I will examine the dataset for Turkey or any other country in the list. For the moment I do not have a codebook but below is a link to the dataset as well as a detailed description:
(http://noracook.io/Books/MachineLearning/deeplearningcookbook.pdf)?
https://www.kaggle.com/javadzabihi/happiness-2017-visualization-prediction/report?select=2017.csv
This dataset gives the happiness rank and happiness score of 155 countries around the world based on seven factors (including family, life expectancy, economy, generosity, trust in government, freedom, and dystopia residual) that provides us the happiness score and the higher the happiness score, the lower the happiness rank. So, it is evident that the higher value of each of these seven factors means the level of happiness is higher. We can define the meaning of these factors as the extent to which these factors lead to happiness. Dystopia is the opposite of utopia and has the lowest happiness level. Dystopia will be considered as a reference for other countries to show how far they are from being the poorest country regarding happiness level.

Since the 1960s, scientific disciplines have researched happiness, to determine how humans can live happier lives. The scientific pursuit of positive emotion and happiness is the pillar of positive psychology, first proposed in 1998 by Martin E. P. Seligman. 

The analysis performs data profiling and analysis for possible action. What countries or regions rank the highest in overall happiness and each of the six factors contributing to happiness? How did country ranks or scores change between the 2015 and 2016 as well as the 2016 and 2017 reports? Did any country experience a significant increase or decrease in happiness?
The purpose of this project is to analyze the dataset by completing the following parts of the report as follows:
•	Cleaning
•	Visualization
•	Prediction etc.
What countries or regions rank the highest in overall happiness and each of the six factors contributing to happiness? How did country ranks or scores change between the 2015 and 2016 as well as the 2016 and 2017 reports? Did any country experience a significant increase or decrease in happiness?
What Method?
Perform exploratory data analysis for examining the data before model selection. The methods I will use are Random Forest, Linear Regression, etc. The Random forest consists of many individual decision trees that operate as an ensemble. Each tree in the random forest spits out a class prediction and the class with the most votes becomes our model’s prediction.
I am planning to do my development on the Jupiter notebook using python. I will be using the notebook as well for the pseudocode. Python packages will be used for developing features, analysis, and testing.
Potential Issues?
The challenges I do expect having maybe the issue to clean the dataset or having a bad one.  An inaccurate dataset could cause the project to go off schedule. Also, the technology issue (internet connection issue) cannot be ignored for delaying a project.
Concluding Remarks
The report gains global recognition because governments, organizations, and civil society gradually use happiness indicators to inform their policy-making decisions. Leading experts across fields such as economics, psychology, survey analysis, national statistics, health, public policy describe how measurements of well-being can assess the progress of nations. The reports review the state of happiness in the world today and show how the new science of happiness explains personal and national variations in happiness.
Reference:
1.	https://www.kaggle.com/unsdsn/world-happiness
2.	https://www.kaggle.com/pinarkaya/world-happiness-eda-visualization-ml/data#Linear-Regression
3.	https://www.kaggle.com/pinarkaya/world-happiness-eda-visualization-ml#Random-Forest
4.	https://www.kaggle.com/unsdsn/world-happiness?select=2019.csv
5.	https://www.kaggle.com/dhanyajothimani/basic-visualization-and-clustering-in-python/data
6.	https://www.kaggle.com/pinarkaya/world-happiness-eda-visualization-ml/data#2019-Data
7.	https://www.kaggle.com/raenish/cheatsheet-70-altair-plots
8.	https://www.kaggle.com/residentmario/data-types-and-missing-values
9.	https://towardsdatascience.com/understanding-random-forest-58381e0602d2
10.	https://en.wikipedia.org/wiki/The_Happiness_Hypothesis
11.	https://www.theworldcounts.com/happiness/the-definition-of-happiness-in-psychology
