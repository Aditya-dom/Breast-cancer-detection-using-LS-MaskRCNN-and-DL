import cv2
#import base64

class MySimpleScaler(object):
    def __init__(self):
        self.new_array = None
        
    def preprocess(self, image):
        #img_str_to_bytes = bytes(image, 'utf-8')
        #with open('gs://newcancer1/images/temp.jpg' , 'wb') as tmp:
        #    tmp.write(base64.decodebytes(img_str_to_bytes))

        #image = cv2.imread('gs://newcancer1/images/temp.jpg')
        new_array = cv2.resize(image, (1152, 896))
        return new_array.reshape(-1, 1152, 896, 3)
  
    
  