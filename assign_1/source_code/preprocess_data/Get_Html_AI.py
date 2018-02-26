#!/usr/bin/python3
#Use Wekipediaapi to get the pages from category AI
import wikipediaapi
import os,sys
#define the number of the html pages
fileNum = 0;

#begin the recursion to get the page in AI
wiki_wiki = wikipediaapi.Wikipedia('en',extract_format=wikipediaapi.ExtractFormat.HTML)
cat = wiki_wiki.page("Category:Artificial_intelligence")

#use recursion to get the relative pages
def output_textincategorymembers(categorymembers, level=0, max_level=1):
        for c in categorymembers.values():
            global fileNum
            global filePath
            fileName = "../Data Set/HTML_Page/POS/%d.txt" % fileNum
            fo = open(fileName, "w")
            fo.write(c.text)
            fileNum += 1
            fo.close()
            if c.ns == wikipediaapi.Namespace.CATEGORY and level <= max_level:
                output_textincategorymembers(c.categorymembers, level + 1)

output_textincategorymembers(cat.categorymembers)
