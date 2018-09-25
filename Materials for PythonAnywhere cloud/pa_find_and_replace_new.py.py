#!/usr/bin/python2.7

from tempfile import NamedTemporaryFile
import shutil
import csv


filename = 'edmunds_extraction.csv'
output = 'edmunds_new_output.csv'
tempfile = NamedTemporaryFile(delete=False)

with open(filename, 'rb') as csvFile, tempfile:
    reader = csv.reader(csvFile, delimiter=',', quotechar='"')
    writer = csv.writer(tempfile, delimiter=',', quotechar='"')

    for row in reader:

        #this item is the forum post
        item = row[2]

        with open('models.csv', 'rb') as csvfile:
            read = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row2 in read:
                    #replace
                    row[2] = row[2].lower().replace(row2[1].lower()," " + row2[0].lower() + " ")

        writer.writerow(row)

shutil.move(tempfile.name, output)

