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

""" The output file:

Please Enter the file Name:swade2
Length of the dictionary: 143
Word         Count        
                     
------------------------------
that         13 
the          11 
we           10 
to           8  
here         8  
a            7  
and          6  
of           5  
not          5  
nation       5  
it           5  
have         5  
for          5  
can          5  
this         4  
in           4  
dedicated    4  
who          3  
us           3  
they         3  
so           3  
shall        3  
people       3  
is           3  
great        3  
dead         3  
are          3  
which        2  
what         2  
war          2  
these        2  
rather       2  
our          2  
or           2  
on           2  
new          2  
men          2  
long         2  
living       2  
gave         2  
from         2  
far          2  
devotion     2  
dedicate     2  
conceived    2  
but          2  
be           2  
years        1  
world        1  
work         1  
will         1  
whether      1  
vain         1  
unfinished   1  
under        1  
thus         1  
those        1  
their        1  
testing      1  
task         1  
take         1  
struggled    1  
should       1  
seven        1  
sense        1  
score        1  
say          1  
resting      1  
resolve      1  
remember     1  
remaining    1  
proposition  1  
proper       1  
power        1  
portion      1  
poor         1  
place        1  
perish       1  
now          1  
november     1  
note         1  
nor          1  
nobly        1  
never        1  
might        1  
met          1  
measure      1  
lives        1  
live         1  
little       1  
lincoln      1  
liberty      1  
last         1  
larger       1  
increased    1  
honored      1  
highly       1  
hallow       1  
ground       1  
government   1  
god          1  
full         1  
freedom      1  
four         1  
fought       1  
forth        1  
forget       1  
fitting      1  
final        1  
field        1  
fathers      1  
equal        1  
engaged      1  
endure       1  
earth        1  
do           1  
died         1  
did          1  
detract      1  
created      1  
continent    1  
consecrated  1  
consecrate   1  
come         1  
civil        1  
cause        1  
by           1  
brought      1  
brave        1  
birth        1  
before       1  
battle-field 1  
as           1  
any          1  
altogether   1  
all          1  
ago          1  
advanced     1  
add          1  
abraham      1  
above        1  
19           1  
1863         1  

Process finished with exit code 0
"""
