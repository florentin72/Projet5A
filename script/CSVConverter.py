import numpy
import cv2
import os
from os import listdir
from sklearn.preprocessing import MinMaxScaler


def generate_x_file(image_folders):
    """Generates training set file."""

    output_path = 'data/x.csv'
    data = []

    for image_folder in image_folders:
        image_names = listdir(image_folder)

        for image_name in image_names:
            # Read image as greyscale
            image_path = "/".join([image_folder, image_name])
            print('image_path', image_path)
            greyscale_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Convert values to values between 0 and 1
            sc = MinMaxScaler(feature_range=(0, 1))
            greyscale_image_normalized = sc.fit_transform(greyscale_image)

            # Convert 2d ndarray to a single list
            pixels_1d_array = greyscale_image_normalized.flatten().tolist()

            data.append(pixels_1d_array)

    # Convert the list of lists into a 2d ndarray
    data = numpy.array(data, numpy.float64)

    with open(output_path, 'wb') as output_file:
        numpy.savetxt(output_file, data, delimiter=",")
        print('Generated file ' + os.path.abspath(output_path))


def generate_y_file(image_folders):
    """Generates target values file."""

    output_path = 'data/y.csv'

    with open(output_path, 'w') as output_file:
        classification = 0
        for image_folder in image_folders:
            images_count = len(os.listdir(image_folder))
            line = str(classification) + '\n'
            lines = images_count * line
            output_file.write(lines)
            classification += 1
        print('Generated file ' + os.path.abspath(output_path))


def read_csv(path):
    """Returns data from file as ndarray"""

    return numpy.loadtxt(path, delimiter=",")


def main():
    image_folders = ['data/EnsembleB/Cercles_2_F',
                     'data/EnsembleB/Cercles_3_F',
                     'data/EnsembleB/Cercles_4_F',
                     'data/EnsembleB/Hexagones_3_F',
                     'data/EnsembleB/Triangles_2_F',
                     'data/EnsembleB/Hexagones_2_F',
                     'data/EnsembleB/Losanges_2_F',
                     'data/EnsembleB/Losanges_3_F']

    if not os.path.exists('data'):
        os.mkdir('data')

    generate_x_file(image_folders)
    generate_y_file(image_folders)


if __name__ == '__main__':
    main()
