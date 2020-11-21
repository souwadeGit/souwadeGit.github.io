---
title: "Weather-Introduction-to-programming"
date: 2020-11-21
tags: [data science, data wrangling, messy data]
header:
  image: "/images/perceptron/percept.jpg"
excerpt: "Data Science, Data Wrangling, Messy Data"
mathjax: "true"
---


print('//**************************************************************************************************************************')
print('// DSC510_Final Project-Weather')
print('// Course: DSC510_Introduction to programming-Python')
print('// Author Name: Soukhna Wade  ')
print('// Date:   11/20/2020')
print('//***************************************************************************************************************************\n')

import requests
from pprint import pprint

# Global Variables
# Note that there are restrictions on the "Free API"  Make sure that you don't submit too many requests

apiKey = "c05250b637031a772b7adbfe110350c5"
url = "http://api.openweathermap.org/data/2.5/weather"
headers = {
    'cache-control': "no-cache",
}

# This method will do a weather lookup by city by taking the city as a parameter

def retrieveWeatherByCity(city):
    querystring = {"q": city + ",us", "units": "imperial", "APPID": apiKey}
    try:
        response = requests.request("GET", url, headers=headers, params=querystring)
    except requests.exceptions.RequestException as e:
        print(e)

    parseWeather(response)


# This method will do a weather lookup by zip
# This method takes the zipCode as a parameter
def retrieveWeatherByZip(zipCode):
    querystring = {"zip": zipCode + ",us", "units": "imperial", "APPID": apiKey}
    try:
        response = requests.request("GET", url, headers=headers, params=querystring)
    except requests.exceptions.RequestException as e:
        print(e)

    # print(response.url)
    # print(response.text)

    parseWeather(response)

# The following method will parse the weather data, present that weather data to the user then takes the response as a parameter

def parseWeather(response):
    weatherJSON = response.json()

    # print(weatherJSON) This is a great troubleshooting line which can be quickly enabled to see your JSON
    pressure = weatherJSON['main']['pressure']
    temp = weatherJSON['main']['temp']
    maxTemp = weatherJSON['main']['temp_max']
    minTemp = weatherJSON['main']['temp_min']
    humidity = weatherJSON['main']['humidity']
    clouds = weatherJSON['clouds']['all']

    if (clouds > 75):
        cloudiness = "Full Cloud Cover"
    elif (clouds > 50 and clouds < 75):
        cloudiness = "Partial Cloud Cover"
    else:
        cloudiness = "Minimual Cloud Cover"
    print("\nCurrent Weather Conditions For " + weatherJSON['name'])
    print("Current Temp: " + str(temp) + " degrees")
    print("High Temp: " + str(maxTemp) + " degrees")
    print("Low Temp: " + str(minTemp) + " degrees")
    print("Pressure: " + str(pressure) + "hPa")
    print("Humidity: " + str(humidity) + "%")
    print("Cloud Cover: " + cloudiness)


"""
   This mian method will prompt the user to make a selection between doing a weather lookup based upon city or zipcode
   The main method will then call a method to do the weather lookup based upon whether the user entered a city or zip
"""

def main():
    runApp = True
    # The program should allow the user to run it multiple times.  This is one way to do that.
    while runApp:
        lookupType = input(
            "\nWould you like to lookup weather data by US City or zip code? Enter 1 for US City 2 for zip: ")

        if (lookupType == "1"):
            city = input("Please Enter The City Name: ")
            retrieveWeatherByCity(city)
        elif (lookupType == "2"):
            zipCode = input("Please enter the zip code: ")
            retrieveWeatherByZip(zipCode)
        else:
            print("Enter a valid option.  Please enter 1 to lookup by city.  Enter 2 to lookup by zip code")
        runApp = input("\nWould you like to perform another weather lookup? (Y/N): ").lower()
        if runApp in 'n':
            break

if __name__ == "__main__":
    main()


