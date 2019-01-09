#!/usr/bin/python3.6
#pip3.6 install --user google-cloud
#pip3.6 install --user google-cloud-vision

import xlrd
from google.cloud import vision
import os
import pandas as pd

Application_Credentials = 'Python-03112e22a573.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = Application_Credentials
client = vision.ImageAnnotatorClient()
image = vision.types.Image()

loc = ("image_url.xlsx")
wb = xlrd.open_workbook(loc)
sheet = wb.sheet_by_index(0)
sheet.cell_value(0, 0)
df = pd.DataFrame()
# loop through every url, retreive the image and send to google vision
for i in range(sheet.nrows):
    image_src_temp = sheet.cell_value(i, 0)
    image.source.image_uri = image_src_temp
    response = client.label_detection(image=image)
    labels = response.label_annotations
    l = []
    for label in labels:
        l.append(label.description)
    s = ' '.join(l)
    print("s")
    print(s)
    df = df.append({'URL': image_src_temp, 'Labels': s}, ignore_index=True)
df.to_excel("Insta_withoutcomments1.xlsx",index=False)


#detect_labels_uri('https://scontent-dfw5-1.cdninstagram.com/vp/e2699090d932dfb696ecc8f01bcac7c5/5C4D0CB8/t51.2885-15/e35/42585029_179491772961780_2445579447934310744_n.jpg')