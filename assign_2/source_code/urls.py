import spotlight
import os
import re

def get_urls(file_path):
    '''
    file_path: the input text file path
    return a list of entity URIs of the input file
    '''
    uris = list()
    file_obj = open(file_path, 'r')
    file_content = file_obj.read()
    if file_content.strip() != '':
        try:
            annotation_dicts = spotlight.annotate('http://api.dbpedia-spotlight.org/en/annotate', file_content, confidence=0.4)

            for dic in annotation_dicts:
                value = dic['URI']
                uris.append(value)            
        except spotlight.SpotlightException:
            print("No Resource")
        else:
            print('Success')


    return uris


folder_path = './txt_data/pos'
filenames =sorted([f for f in os.listdir(folder_path) if not f.startswith('.')],  key=lambda f: (int(re.sub('\D','',f)),f)) 

for i, file_name in enumerate(filenames):
    file_path =  os.path.join(folder_path, file_name)
    file_URIs = get_urls(file_path)
    # print(file_URIs)
    URLs = open('./urls_data/cs_urls/%d.txt' % i, 'w')
    for url in file_URIs:
        url = url.encode('ascii', 'ignore').decode('ascii')
        URLs.write(url+ '\n')





