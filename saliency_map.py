import numpy as np
import cv2
import time
import pandas as pd
from cv2 import imread
import os
import tensorflow as tf

tf.flags.DEFINE_string('input_dir', './NIPS2017/images', 'dev_data path')
tf.flags.DEFINE_string('output_dir', './NIPS2017/saliency_maps/', 'generated saliency maps path')
tf.flags.DEFINE_integer('data', 80, 'generated saliency maps path')

tf.flags.DEFINE_string('csv_file', './NIPS2017/images.csv', 'csv_file path')
FLAGS = tf.flags.FLAGS

input_dir = FLAGS.input_dir
output_dir = FLAGS.output_dir
csv = FLAGS.csv_file


def load_images(input_dir):
    """
    Load images from given image directory.
    :param input_dir: Image directory
    :return: Img_names and corresponding images
    """
    images = []
    filenames = []
    dev = pd.read_csv(os.path.join(csv))
    filename2label = {dev.iloc[i]['ImageId'] for i in range(1)}
    for filename in filename2label:
        dir = os.path.join(input_dir, filename + ".png")
        image = imread(dir)
        images.append(image)
        filenames.append(filename)
    images = np.array(images)

    return filenames, images


def euclidean(x, y):
    """
    Calculate Euclidean Distance
    """
    return np.sqrt(np.sum(np.square(x - y)))


def saliency2color(img_width, saliency_map1, i, rgb_colors, color_map):
    """Convert Black&White Salinecy map into color map"""
    saliency_range = 10000  # 128*128*128
    for j in range(img_width):
        if saliency_map1[i][j] > 0:
            ratio = int(min(99, saliency_map1[i][j]) * (saliency_range - 1) // max(map(max, saliency_map1)))
            color_map[i][j] = rgb_colors[ratio]
        else:
            color_map[i][j] = rgb_colors[0]
    return color_map[i]


def generate_saliency(image, box_centers):
    """
    Generate saliency map for an image
    :param image: Input image
    :param box_centers: A set of the center of the feature points extracted by SIFT
    :return: The saliency map of input image
    """
    img_width = image.shape[1]
    img_height = image.shape[0]
    s_map = np.zeros((img_height, img_width))

    sum_res = sum([i[1] for i in box_centers])
    for i in range(len(box_centers)):
        box = box_centers[i]
        for j in range(max(0, int(box[0][0] - box[2])), min(299, int(box[0][0] + box[2]) + 1)):
            for k in range(max(0, int(box[0][1] - box[2])), min(299, int(box[0][1] + box[2]) + 1)):
                coords = [j, k]
                dist = euclidean(np.array(box[0]), np.array(coords))
                if dist <= box[2]:
                    s_map[k][j] += np.log((box[1]/sum_res + 1))

    saliency = s_map * 255 // np.max(s_map)
    return saliency


def check_or_create_dir(directory):
    """Check if directory exists otherwise create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def main():
    start = time.time()
    sift = cv2.xfeatures2d.SURF_create(500)
    filenames, imgs_input = load_images(input_dir)
    check_or_create_dir(output_dir)

    count = 0
    for img in imgs_input:
        print(count)
        img = np.array(img)

        kp, des = sift.detectAndCompute(img, None)
        location = []
        for k in kp:
            location.append((k.pt, k.response, k.size))

        saliency_map = generate_saliency(img, location)
        cv2.imwrite(os.path.join(output_dir, filenames[count]+ ".png"), saliency_map)

        count += 1
    print("total time is {}".format(time.time() - start))


if __name__ == '__main__':
    main()
