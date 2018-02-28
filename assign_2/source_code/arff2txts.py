import spotlight
import re
#Read the AI and BIO documents from text file
with open('./data/cs.txt') as file_object:
    contents = file_object.read()

pattern = re.compile('\'(.*)\'')
# print (pattern.findall(contents))
for i, text in enumerate(pattern.findall(contents)):
    file_object = open('./data/cs/%d.txt' %i, 'w')
    file_object.write(text)
