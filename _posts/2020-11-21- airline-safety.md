---
title: "Airline Safety"
date: 2020-11-21
tags: [data science, data wrangling, messy data]
header:
  image: "/images/perceptron/percept.jpg"
excerpt: "Data Science, Data Wrangling, Messy Data"
mathjax: "true"
---



### DSC640-T301 Data Presentation & Visualization (2207-1) 
###  Weeks 1 & 2  /  1.3 Project Task 1: Dashboard  
  

data <- read.csv("C:/Users/Soukhna/Desktop/DSC640_06012020/DSC 640_project/data-master/data-master/airline-safety/airline-safety.csv")
data
View(data)
colnames(data)             # shows the name of each column in the data frame
#head(data)                 # shows the first 6 rows of the data frame ## look at the first several rows of the data
#tail(data)



# Scatter plots and jittering (1)

# Shown in the viewer:
library(ggplot2)
ggplot(data, aes(x = airline, y = fatalities_85_99)) +
  geom_point()




nrow(data)         
ncol(data)


dim(data)                  #shows the dimensions of the data frame by row and column
str(data)                  # shows the structure of the data frame
summary(data)              # provides summary statistics on the columns of the data frame
colnames(data)             # shows the name of each column in the data frame
head(data)                 # shows the first 6 rows of the data frame ## look at the first several rows of the data
tail(data)                 # shows the last 6 rows of the data frame
#View(data)                # shows a spreadsheet-like display of the entire data frame
#rownames(data)
#colnames(data)

attach(data)              #attach the data frame to the environment

##Explore the data



library(dplyr)
glimpse(data) 

#install.packages("plotly") 
#library(plotly) 


library(plotly)

fig <- plot_ly(
  x = c("giraffes", "orangutans", "monkeys"),
  y = c(20, 14, 23),
  name = "SF Zoo",
  type = "bar"
)

fig


library(plotly)

Animals <- c("giraffes", "orangutans", "monkeys")
SF_Zoo <- c(20, 14, 23)
LA_Zoo <- c(12, 18, 29)
data <- data.frame(Animals, SF_Zoo, LA_Zoo)

fig <- plot_ly(data, x = ~Animals, y = ~SF_Zoo, type = 'bar', name = 'SF Zoo')
fig <- fig %>% add_trace(y = ~LA_Zoo, name = 'LA Zoo')
fig <- fig %>% layout(yaxis = list(title = 'Count'), barmode = 'group')

fig


