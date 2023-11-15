# Breast-Cancer-Lump-Segmentation-MaskRCNN
Breast Cancer lump segmentation using mask RCNN. Implementation on tensorflow with CBIS-DDSM dataset. 


## Developer's Note
Delete this readme and make a new one once the development is done. 
Mark it yours Buddy!!

### Image Preprocessing

#### Source
This dataset was downloaded from CBIS DDSM website [here](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM), it is really not that complicated to download this dataset, the download documentation is on their website. 

#### Conversion
This dataset was in DCM or dicom format "Digital Imaging and Communications in Medicine (DICOM) is the standard for the communication and management of medical imaging information and related data". From that we converted this script into png format using imagemagic(mogrify), incorporating the imagemagic into some bash scripts we were able to convert the entire dataset into png, this is really not necessary to mention but the framework and research paper I have followed, made it so that we had to get the patches of mammograms and region of interests, and for taking our each patch and associating each patch with its ROI(region-of-interest) we had to come up with a logic which was, that each mammogram and ROI has been mentioned already in a csv that comes with the dataset from CBIS, using that naming convention we named each mammogram and ROI, and followed that naming convention to zip and then classify each image with its respective case. This just makes it easy for us to take out the patches and keep them associated with the cases in that dataset. 

#### Patches
As mentioned before using a quite complicated logic each patch was taken out from the mammogram and was associated with its ROI. 

## Cheers!!