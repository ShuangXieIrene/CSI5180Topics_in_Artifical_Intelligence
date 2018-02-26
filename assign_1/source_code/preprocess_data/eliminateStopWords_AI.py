# -*- coding: UTF-8 –*-
#Eliminate the stop words
import urllib2
import re
import string
import operator

global i

def isCommon(ngram):
    stopName = "/home/zehua/Desktop/Assignment1/Data_Clean/stopwords.txt"
    f = open(stopName)
    stopwordsList = []
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        stopwordsList.append(line)
    commonWords = stopwordsList
    if ngram in commonWords:
        return True
    else:
        return False

def cleanInput():
    for i in range(0,5000):
        fileName = "/home/zehua/Desktop/Assignment1/Data_Set/AfterParse/POS/%d" % i
        stopName = "/home/zehua/Desktop/Assignment1/Data_Set/Eliminate_StopWord/POS/%d.txt" % i
        input  = open(fileName).read()
        #print(input)
        cleanInput = ''
        input = input.split(' ')

        for item in input:
            item = item.strip(string.punctuation)

            if(isCommon(item)):
                pass
            else:
                if len(item) > 1 or (item.lower()=='a' or item.lower() == 'i'):
                    cleanInput = cleanInput + ' ' + item

        #print(cleanInput)
        f = open(stopName,'w')
        f.write(cleanInput)
        f.close

cleanInput()
