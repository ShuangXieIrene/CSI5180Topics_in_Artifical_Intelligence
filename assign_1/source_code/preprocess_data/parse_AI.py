#-*- coding:utf-8 -*-
#Use beautiful Soup to Clean The irrelevant symbol and blank
import nltk
import sys
import re
from urllib import urlopen
from bs4 import BeautifulSoup
reload(sys)
sys.setdefaultencoding( "utf-8" )

for i in range(0,5000):
    url='file://///home/zehua/Desktop/Assignment1/Data_Set/HTML_Page/POS/%d.txt' % i
    html_doc = urlopen(url).read()
    soup = BeautifulSoup(html_doc,"lxml")

    fo = open("/home/zehua/Desktop/Assignment1/Data_Set/Cleaned_Page/%d.txt" % i, "w")
    fo.write(soup.getText())
    fo.close()
    fo = open("/home/zehua/Desktop/Assignment1/Data_Set/Cleaned_Page/%d.txt" % i, "r")
    lines = fo.readlines()
    fo.close()

    fo = open("/home/zehua/Desktop/Assignment1/Data_Set/CleanedMore_Page/%d" %i, "w")
    for line in lines:
        line = line.decode('utf8')
        line = re.sub('\n+', " ", line).lower()
        line = re.sub(' +', " ", line)
        line = re.sub("[0-9\[\`\~\!\@\#\$\^\&\*\(\)\=\|\{\}\'\:\;\'\,\[\]\.\<\>\/\?\~\！\@\#\\\&\*\%]", "", line)
        line = re.sub('\[[0-9]*\]', "", line)
        line = line.replace('+', '').replace('.', '').replace(',', '').replace(':', '').replace(';','').replace("(",'').replace(")",'').replace('/','').replace('=','').replace(']','').replace('[','')
        line = line.replace('"','').replace("'",'').replace('-','').replace('`','').replace('+','').replace('×','').replace('&','').replace('…','')
        line = re.sub(' +', " ", line)
        line = re.sub("(^|\W)\d+($|\W)", " ", line)
        line = re.sub(' +', " ", line)
        line = re.sub('\n+', " ", line).lower()
        fo.write(line)
    fo.close()
