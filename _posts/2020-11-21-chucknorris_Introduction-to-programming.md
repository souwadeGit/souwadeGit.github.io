---
title: "chucknorris_Introduction-to-programming"
date: 2020-09-13
tags: [data science, data wrangling, messy data]
header:
  image: "/images/happiness.jpg"
excerpt: "Data Science, Data Wrangling, Messy Data"
mathjax: "true"
---



print('//************************************************************************************************************************')
print('// File:   Program DSC 510')
print('// Name:   Soukhna Wade')
print('// Date:   11/20/2020')
print('// Course: DSC510_Introduction to Python')
print('// Desc:   This program uses an open API to obtain data for the end user.')
print('//         This program will receive a JSON response which includes various pieces of data. ')
print('//         This program displays for the user the data associated with the value key. ')
print('//         This program generates “pretty” output by dumping a bunch of data to the screen. ')
print('// Usage:  Uses the Request library to make a GET request of the following API: Chuck Norris Jokes.')
print('//         Parse the JSON data to obtain the “value” key.')
print('//*************************************************************************************************************************')

import requests

print("WELCOME TO THE JOKE ERA!")
print("PLEASE BE AWARE THAT BY PRESSING Y OR y, YOU WILL CONTINUE TO PLAY, AND ANY OTHER KEY TO QUIT THE JOKE ERA.")
# api-endpoint
URL = " https://api.chucknorris.io/jokes/random"
# sending get request and saving the response as response object
try:

    while True:
        res = requests.get(url=URL)
        # extracting data in json format
        data = res.json()
        joke = data['value']
        print('The joke is {}'.format(joke))
        y = input("DO YOU WANT TO CONTINUE?: ")
        if y.lower() == "y":
            continue
        else:
            print('END OF THE JOKE ERA')
            break

except requests.exceptions.Timeout:
    # Maybe set up for a retry, or continue in a retry loop
    print("There is timeout error")
except requests.exceptions.TooManyRedirects:
    # Tell the user their URL was bad and try a different one
    print("There is bad request . Please check your URL")
except requests.exceptions.HTTPError as errh:
    print("Http Error:", errh)
except requests.exceptions.RequestException as exc:
    # catastrophic error. bail.
    print("Error:", exc)
    sys.exit(1)

