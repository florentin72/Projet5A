from PIL import Image
import os
from os import listdir
import sys


def resize_image(img_path, new_img_path, new_width, new_height):
    img = Image.open(img_path)
    img = img.resize((new_width, new_height), Image.ANTIALIAS)
    img.save(new_img_path)


def main():
    if len(sys.argv) == 2:
        # Take script first argument if exists as images_path
        images_path = sys.argv[1]
    else:
        # Use hardcoded images_path
        images_path = "data/EnsembleA/Losanges/Losanges_2_F"

    print('images_path', os.path.abspath(images_path))
    image_names = listdir(images_path)

    # Create output folder in current directory if it does not already exist
    basename = os.path.basename(os.path.normpath(images_path))
    output_folder = "data/resized/" + basename
    print('output_folder', os.path.abspath(output_folder))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_name in image_names:
        image_path = "/".join([images_path, image_name])
        output_path = "/".join([output_folder, image_name])

        resize_image(image_path, output_path, 32, 24)


if __name__ == '__main__':
    main()
