# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # 基礎1

3+3

(50-5*6)/4

5/6

5.0/6

5//6

float(5)/6

import math

math.sqrt(36)

from math import sqrt

sqrt(36)

volume = 10

volume

red = 5

red*volume

'Hello world!'

"Hello world!"

print('こんにちは')

print(' She said "hello"')

phrase = 'hello'

phrase

print(phrase)

print('My volume is {}'.format(volume))

phrase = 'hello ' + 'world!'

phrase

cities = ['NYC','LA','SF']

cities

cities[0]

cities[1]

cities[-1]

cities.append('CHI')

cities

range(10)

list(range(10))

list(range(3,8))

list(range(0,20,2))

random_list = ['hello',76,'today',4.3]

type('hello')

type(76)

type(4.3)

type(random_list)

len(random_list)

# # 基礎2

cities = ['NY','LA','SF']

for city in cities:
    print(city)

for city in cities:
    phrase = 'I love ' + city
    print(phrase)

for n in range(1,10):
    print('The inverse of',n,'is',1.0/n)

for letter in 'Hello':
    print(letter)

cities[0:2]

if city == 'NY':
    print('Party')
else:
    print('Work')

city = 'NY'

1 == 2

2 == 2

3>4

4<5

1<=2

1!=2

1>=0

'Hello' == 'hello'

'Hello' == 'Hello'

# # 基礎3

cities = ['NY','LA','SF']

city = cities[0]

city

for city in cities:
    if city == 'NY':
        print('Party')
    elif city == 'LA':
        print("It's hot here.")
    else:
        print("Where am I?")

t = (1,2,3)

t.append(2)

my_list = [1,2,3]

my_dict = {'Taro':22,'Yoko':12}

my_dict['Taro']

len(my_dict)


def adder(x,y):
    '''これはxとyを足す関数です'''
    answer = x + y
    return answer


adder(5,10)


