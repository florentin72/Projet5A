""" Greyscale Converter

This script takes a folder path in parameter. All images in the folder will be copied as greyscale images to the data
folder.

e.g. "python GreyscaleConverter.py data\EnsembleA\Losanges\Losanges_2_F" will copy all images from that folder to
data\Greyscale_Losanges_2_F.
"""


import os
from os import listdir

import cv2
import sys


def visualize_image(window_name, img):
    cv2.moveWindow(window_name, 40, 30)
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    if len(sys.argv) == 2:
        # Take script first argument if exists as images_path
        images_path = sys.argv[1]
    else:
        # Use hardcoded images_path
        images_path = "data\EnsembleA\Losanges\Losanges_2_F"

    print('images_path', os.path.abspath(images_path))
    image_names = listdir(images_path)

    # Create output folder in current directory if it does not already exist
    basename = os.path.basename(os.path.normpath(images_path))
    output_folder = "data\Greyscale_" + basename
    print('output_folder', os.path.abspath(output_folder))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_name in image_names:
        # Read image as greyscale
        image_path = "/".join([images_path, image_name])
        greyscale_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # visualize_image(image_name, greyscale_image)

        # Write greyscale image in output folder
        output_path = "/".join([output_folder, image_name])
        cv2.imwrite(output_path, greyscale_image)


if __name__ == '__main__':
    main()
