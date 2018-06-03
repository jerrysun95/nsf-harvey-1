'''
This script creates a directory for each date and processes all links for the first cmd line argument
Created by Matthew Johnson 05/26/2018
''' 

import os
import csv
from urllib2 import urlopen
import urllib
import sys

date = sys.argv[1]
# get dictionary of <date, <list of links>>
d = {}
with open('Media_Links.txt', 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    #for row in reader:
    d = {row[0]: row[1].split(' ') for row in reader}



print "Processing date " + str(date) 

# make new directory
try: 
    os.makedirs(date)
except OSError:
    if not os.path.isdir(date):
        raise       
links = d.get(date)

# filter out null links 
links = filter(lambda x: x != 'NULL', links)


# get images from URLs, then save to directory
count=1
for link in links: 
    print "processing link " + str(link)

    res = urllib.urlopen(link)
    filename = date + "-" + str(count) + ".jpg"
    fullfilename = os.path.join(date + "/", filename)
    out = open(fullfilename, "wb")
    out.write(res.read())
    out.close()
    count = count  + 1




