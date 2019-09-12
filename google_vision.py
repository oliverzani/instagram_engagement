
'''
This script will send each instagram image through google vision's API and return the output in an excel file

Inputs:
    
    1. Replace the value of 'input_file' on line 29 with the name of the output file from collect_from_insta.py
        
    2. Replace the value of 'output_file' on line 30 with the name of your desired output file
                        
Outputs:

    1. The output file with each images Google Vision output and corresponding URL
'''

#Install Google Vision if necessary
#pip install google-cloud
#pip install google-cloud-vision

#Packages to import
import xlrd
from google.cloud import vision
import os
import pandas as pd

os.chdir('c:/users/olive/documents/class/medium')

#Inputs
input_file = "marke_miller.xlsx"
output_file = 'google_marke_miller.xlsx'

#Google Application Credentials
Application_Credentials = 'google_vision_parameters.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = Application_Credentials
client = vision.ImageAnnotatorClient()
image = vision.types.Image()

#Setting up excel workbook to be used
wb = xlrd.open_workbook(input_file)
sheet = wb.sheet_by_index(0)
df = pd.DataFrame()

# loop through every url, retreive the image and send to google vision
for i in range(1,sheet.nrows):
    image_src_temp = sheet.cell_value(i, 4)
    image.source.image_uri = image_src_temp
    response = client.label_detection(image=image)
    labels = response.label_annotations
    l = []
    for label in labels:
        l.append(label.description)
    s = ' '.join(l)
    df = df.append({'URL': image_src_temp, 'Labels': s}, ignore_index=True)
    
df.to_excel(output_file, index=False)
print("Written to " + output_file)



