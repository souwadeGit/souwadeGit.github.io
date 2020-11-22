---
title: "Gettysburg-Introduction-to-programming"
date: 2020-11-21
tags: [data science, data wrangling, messy data]
header:
  image: "/images/perceptron/percept.jpg"
excerpt: "Data Science, Data Wrangling, Messy Data"
mathjax: "true"
---

print('//************************************************************************************************************************')
print('// File:   Program1: open/calculate words')
print('// Name:   Soukhna Wade')
print('// Date:   11/20/2020')
print('// Course: DSC510_Introduction to Python')
print('// Desc:   This program performs three essential operations')
print('//         It will process this .txt file: Gettysburg.txt. ')
print('//         Nicely print the output, in this case from high to low frequency n/  You should use string formatting for this. (See discussion 8.3)')
print('//         Calculate the total words, and output the number of occurrences of each word in the file.')
print('// Usage:  Program will use string formatting')
print('//         Click the link to download the text file named Gettysburg.txt')
print('//*************************************************************************************************************************')


import string
from re import split

def pretty_print(word_dict):              #Print nicely from highest to lowest frequency
    value_key_list = []
    for key, val in word_dict.items():
        value_key_list.append((val, key))
        # sort method sorts on list's first element, the frequency.
    value_key_list.sort(reverse=True)                  # Reverse to get biggest first
    print('{:13s}{:13s}'.format('Word', 'Count'))
    print(' ' * 21)
    print("---------------------------")
    for val, key in value_key_list:
        print('{:12s} {:<3d}'.format(key, val))

def process_line(line, word_dict):
    line = line.strip()                            # Remove the leading spaces and newline character
    word_list = line.split(" ")                    # Split the line into word_list
    for word in word_list:                         # Loop through each word of the file
        if word != '--':                           # ignore the '--' in the file
            word = word.lower()                    #convert the characters in line to lowercase to avoid case mismatch
            word = word.strip()
            word = word.strip(string.punctuation)
            add_word(word, word_dict)              # add_word(word, word_dict)

def add_word(word, word_dict):                     #  add_word: Add each word to the dictionary
    if word in word_dict:                          # Check if the word is already in dictionary
        word_dict[word] += 1                       # Increment count of word by 1
    else:
        if word != '':                              # check if word is not an empty space
            word_dict[word] = 1

def main():
    word_count_dict = {}
    text = open('gettysburg.txt', 'r')              # Open a file for reading:
    for line in text:
        process_line(line, word_count_dict)
    print('Length of the dictionary:', len(word_count_dict))
    pretty_print(word_count_dict)
if __name__ == "__main__":
    main()                                            # execute only if run as a script


