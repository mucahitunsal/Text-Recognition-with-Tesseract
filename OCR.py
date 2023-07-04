import cv2
import pytesseract
from preprocessclass import ImagePreprocessor
import random
import os

try:
    os.mkdir("../Saved_Texts")
except OSError as error:
    print(error)


filter = ImagePreprocessor()

img = cv2.imread('../Test_Images/tesT_img.jpeg')

def preprocess(frame):
    rescaled = filter.rescale(frame, 1.25)
    gray = filter.gray_filter(rescaled)
    # straight = filter.straighten_image(gray)
    th = filter.thresholding(gray, threshold_value=180)
    no_noise = filter.remove_noise(th)
    thin_font = filter.thin_font(no_noise)

    return thin_font


pre_processed = preprocess(img)


def get_text(frame):
    return pytesseract.image_to_string(frame)


text = get_text(img)

# with open("../Saved_Texts/output_{}.txt".format(str(random.random())), "w", encoding="utf8") as file:
    # file.write(text)
    # print("Dosya Kaydedildi.")

title = ""
if len(text) > 13:
    title += text[:13]
else:
    title += text

with open("../Saved_Texts/" + title + ".txt", "w", encoding="utf8") as file:
    file.write(text)
    print("Dosya Kaydedildi.")

cv2.destroyAllWindows()
