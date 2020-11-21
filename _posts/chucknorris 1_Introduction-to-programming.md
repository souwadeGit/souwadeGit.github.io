---
title: "chucknorris 1_Introduction-to-programming"
date: 2020-09-13
tags: [data science, data wrangling, messy data]
header:
  image: "/images/happiness.jpg"
excerpt: "Data Science, Data Wrangling, Messy Data"
mathjax: "true"
---

# DSC-510_Introduction-to-programming 
 
## Chuck Norris Jokes -Program


We have already looked at several examples of API integration from a Python perspective and now we are going to write a program that uses an open API to obtain data for the end user.
•	Create a program which uses the Request library to make a GET request of the following API: Chuck Norris Jokes.
•	The program will receive a JSON response which includes various pieces of data. You should parse the JSON data to obtain the “value” key. The data associated with the value key should be displayed for the user (i.e., the joke).
•	Your program should allow the user to request a Chuck Norris joke as many times as they would like. You should make sure that your program does error checking at this point. If you ask the user to enter “Y” and they enter y, is that ok? Does it fail? If it fails, display a message for the user. There are other ways to handle this. Think about included string functions you might be able to call.
•	This program must include a header as in previous weeks.
•	This program must include a welcome message for the user.
•	This program must generate “pretty” output. Simply dumping a bunch of data to the screen with no context doesn’t represent “pretty.

From <https://api.chucknorris.io/jokes/random> 
Program use: Python
