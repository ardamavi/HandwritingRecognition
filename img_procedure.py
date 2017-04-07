# Arda Mavi
import numpy
from PIL import Image

def getImg(img_dir):
    try:
        # Get grayscale image matrix from directory:
        img = Image.open(img_dir).convert('L')
    except:
        return None
    # If image size not 25x25, resize:
    if img.size != (25, 25):
        img = img.resize((25, 25))
    # Return image matrix list:
    img = numpy.array(img).tolist()
    img_list = []
    for line in img:
        for value in line:
            img_list.append(value)
    return img_list

def getImgDir(argumans):
    # We get the image directory:
    if len(argumans) < 2:
        return None
    return argumans[1]
