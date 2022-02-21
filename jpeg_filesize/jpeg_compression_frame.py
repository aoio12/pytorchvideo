import glob
import imgaug.augmenters as iaa
import imageio
import imgaug as ia
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

files = glob.glob(
    "/mnt/HDD10TB-2/ohtani//dataset/frame_jpeg/jpeg_compression_frame_2/NA/*.jpg")
# print(len(files))
file_path = '/mnt/HDD10TB-2/ohtani//dataset/frame_jpeg/jpeg_compression_frame_2/0/'
i = 1
# print(files)
for file in files:

    # image = np.array(Image.open(file))
    # print(Image.open(file).mode)
    # # image=imageio.imread(file)
    # # image=np.asarray(image)
    # aug = iaa.JpegCompression(compression=80)
    # img_jpeg = aug.augment_image(image)
    # # img_jpeg = np.array(img_jpeg.astype(np.uint8))
    # Image.fromarray(np.uint8(img_jpeg))

    image = imageio.imread(file)
    image = np.asarray(image)
    aug = iaa.JpegCompression(compression=0)
    img_jpeg = aug.augment_image(image)
    plt.imsave(file_path + '{0:06d}.jpg'.format(i), img_jpeg)
    i += 1
