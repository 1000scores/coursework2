import os
from PIL import Image

img = Image.open('data/rtsd/autosave10_10_2012_12_06_35_4.jpg').convert('RGB')

print(img)