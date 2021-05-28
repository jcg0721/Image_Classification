import time
from os import listdir
from os.path import isfile, join
from PIL import Image
from random import shuffle


IMAGE_SIZE = 224
IMAGE_PATH = "C:/Users/ChangGwon/Desktop/kadid700k/kadis700k/"
NORMAL_DIR = "normal1500"
DARK_DIR = "dark750"
BRIGHT_DIR = "bright750"
BLUR_DIR = "blur512"
RESIZE_PATH = "ResizeImage6000\\"
LABELING_FILE = "labeling6000.txt"
RESIZE = (IMAGE_SIZE, IMAGE_SIZE)


labeing_file = open(LABELING_FILE, 'w')
normal_file_names = [f for f in listdir(IMAGE_PATH + NORMAL_DIR) if isfile(join(IMAGE_PATH + NORMAL_DIR, f))] #0
dark_file_names = [f for f in listdir(IMAGE_PATH + DARK_DIR) if isfile(join(IMAGE_PATH + DARK_DIR, f))] #1
bright_file_names = [f for f in listdir(IMAGE_PATH + BRIGHT_DIR) if isfile(join(IMAGE_PATH + BRIGHT_DIR, f))] #2
blur_file_names = [f for f in listdir(IMAGE_PATH + BLUR_DIR) if isfile(join(IMAGE_PATH + BLUR_DIR, f))] #3


def Resize(file_names, dir, label):
    fileList = []
    for i in range(len(file_names)):
        image = Image.open(IMAGE_PATH + dir + "\\" + file_names[i], 'r')

        if image.size[0] > image.size[1]:
            box = ((image.size[0] - image.size[1]) / 2, 0, (image.size[0] + image.size[1]) / 2, image.size[1])
            crop_Image = image.crop(box)

        else:
            box = (0, (image.size[1] - image.size[0]) / 2, image.size[0], (image.size[1] + image.size[0]) / 2)
            crop_Image = image.crop(box)

        resize_Image = crop_Image.resize(RESIZE)

        imageName = str(int(round(time.time() * 1000))) + ".jpg"
        resize_Image.save(RESIZE_PATH + imageName)

        fileList.append(imageName + " " + label)

    return fileList


normal = Resize(normal_file_names, NORMAL_DIR, "0")
dark = Resize(dark_file_names, DARK_DIR, "1")
bright = Resize(bright_file_names, BRIGHT_DIR, "2")
blur = Resize(blur_file_names, BLUR_DIR, "3")


total = normal
total.extend(dark)
total.extend(bright)
total.extend(blur)
shuffle(total)


for list in total:
    labeing_file.write(list + "\n")


