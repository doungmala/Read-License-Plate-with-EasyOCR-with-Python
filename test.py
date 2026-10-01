import easyocr
import os
import cv2

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
reader = easyocr.Reader(['th', 'en'], gpu=False)
print('reader',reader)
result = reader.readtext('1123k.jpg', detail=0)
print(result)
print(result[0])

img = cv2.imread("1123k.jpg")
# fetching the dimensions
wid = img.shape[1]
hgt = img.shape[0]
# displaying the dimensions
print(str(wid) + "x" + str(hgt))

