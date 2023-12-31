

import pandas as pd

CC = 'CC'
MLO = 'MLO'

csv = pd.read_csv('csv/calc_case_description_test_set.csv')
imagefile = 'Calc-Test_P_02275_LEFT_CC_FLIP_65'
#imagefile = 'kazim'
if(CC in imagefile):    
    imagefile, _ = imagefile.split('CC')
    imagefile+='CC'
elif(MLO in imagefile):
    imagefile, _ = imagefile.split('MLO')
    imagefile+='MLO'
else:
    pass
print(imagefile)

#if(csv['image file path'].str.find(imagefile) != -1):
#    maskfile = csv['ROI mask file path']
#img = 'Calc-Test_P_01562_LEFT_MLO'
maskfile = csv[csv['image file path'].str.contains(imagefile)]
maskfile = maskfile['ROI mask file path']
maskfile = maskfile.to_frame()
maskfile = maskfile.iloc[0][0]
maskfile = maskfile.split('/')
print(maskfile[0])

maskfile = csv[csv['image file path'].str.contains(imagefile)]
maskfile = maskfile['ROI mask file path'].to_frame().iloc[0][0].split('/')
print("REAL ",maskfile[0])

pathology = csv[csv['image file path'].str.contains(imagefile)]
pathology = pathology['pathology'].to_frame().iloc[0][0]
print('Thats pathology ', pathology[0])