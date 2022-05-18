# -*- coding: utf-8 -*
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from nets import vgg, inception, resnet_v2
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
slim = tf.contrib.slim
tf.flags.DEFINE_integer('noise', 16, 'l_inf size of noise')
tf.flags.DEFINE_string('input_dir', './datasets/images', 'dev_data path')
tf.flags.DEFINE_string('saliency_maps', './datasets/saliency_maps', 'saliency_map path')
tf.flags.DEFINE_string('output_dir', './results/ensemble', 'output_file path ')
tf.flags.DEFINE_string('csv_file', './datasets/images.csv', 'csv_file path')
tf.flags.DEFINE_integer('batch_size', 25, 'How many images process at one time.')

tf.flags.DEFINE_string('checkpoint_path', './checkpoints/', 'Directory of checkpoint path')
tf.flags.DEFINE_integer('image_width', 224, 'Width of vgg images.')
tf.flags.DEFINE_integer('image_height', 224, 'Height of vgg images.')
tf.flags.DEFINE_float('momentum', 1.0, 'momentum')
tf.flags.DEFINE_integer('N', 7, 'number of angles')

FLAGS = tf.flags.FLAGS
model_checkpoint_map = {'inception_v4': os.path.join(FLAGS.checkpoint_path, 'inception_v4.ckpt'),
                        'resnet_v2_152': os.path.join(FLAGS.checkpoint_path, 'resnet_v2_152/resnet_v2_152.ckpt'),
                        'inception_resnet_v2': os.path.join(FLAGS.checkpoint_path, 'inception_resnet_v2.ckpt'),
                        'vgg_19': os.path.join(FLAGS.checkpoint_path, 'vgg_19.ckpt')}
max_epsilon = FLAGS.noise
num_angles = int(FLAGS.N)
batch_size = FLAGS.batch_size
momentum = FLAGS.momentum
inner_iters = 5
last_iters = 10
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94


def noise_fusing(input_tensor, b_size, filter_w):
    """
    For each image, convolve noises from different models with gaussian kernel
    :param input_tensor: the noises to fuse
    :param b_size: batch-size
    :param filter_w: gaussian kernel
    :return: fused noise
    """
    input_tensor = tf.transpose(input_tensor, perm=[1, 0, 2, 3, 4])
    output = []
    # for each image, convolve noises from different models with gaussian kernel
    for i in range(b_size):
        noise = tf.transpose(input_tensor[i], perm=[3, 1, 2, 0])
        noise = tf.nn.conv2d(input=noise, filter=filter_w, strides=[1, 1, 1, 1], padding='SAME')
        noise = tf.concat(values=[noise[0], noise[1], noise[2]], axis=2)
        output.append(noise)
    output = tf.convert_to_tensor(output)

    return output


def rad(x):
    return x * np.pi / 180


def perspective_trandformation(w, h, angle_x, angle_y):
    """
    Calculate perspective projection operator.
    :param w: image's width
    :param h: image's height
    :param angle_x: projection angle for x-axis
    :param angle_y: projection angle for y-axis
    :return: Perspective projection operator.
    """
    angle_z = 0
    fov = 42
    z = np.sqrt(w ** 2 + h ** 2) / 2 / np.tan(rad(fov / 2))

    rx = np.array([[1, 0, 0, 0],
                   [0, np.cos(rad(angle_x)), -np.sin(rad(angle_x)), 0],
                   [0, -np.sin(rad(angle_x)), np.cos(rad(angle_x)), 0],
                   [0, 0, 0, 1]], np.float32)

    ry = np.array([[np.cos(rad(angle_y)), 0, np.sin(rad(angle_y)), 0],
                   [0, 1, 0, 0],
                   [-np.sin(rad(angle_y)), 0, np.cos(rad(angle_y)), 0],
                   [0, 0, 0, 1]], np.float32)

    rz = np.array([[np.cos(rad(angle_z)), np.sin(rad(angle_z)), 0, 0],
                   [-np.sin(rad(angle_z)), np.cos(rad(angle_z)), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], np.float32)

    r = rx.dot(ry).dot(rz)

    pcenter = np.array([h / 2, w / 2, 0, 0], np.float32)

    p1 = np.array([0, 0, 0, 0], np.float32) - pcenter
    p2 = np.array([w, 0, 0, 0], np.float32) - pcenter
    p3 = np.array([0, h, 0, 0], np.float32) - pcenter
    p4 = np.array([w, h, 0, 0], np.float32) - pcenter

    dst1 = r.dot(p1)
    dst2 = r.dot(p2)
    dst3 = r.dot(p3)
    dst4 = r.dot(p4)
    list_dst = [dst1, dst2, dst3, dst4]

    org = np.array([[0, 0],
                    [w, 0],
                    [0, h],
                    [w, h]], np.float32)

    dst = np.zeros((4, 2), np.float32)

    for i in range(4):
        dst[i, 0] = list_dst[i][0] * z / (z - list_dst[i][2]) + pcenter[0]
        dst[i, 1] = list_dst[i][1] * z / (z - list_dst[i][2]) + pcenter[1]

    warpR = cv2.getPerspectiveTransform(org, dst)
    return warpR


def preprocess_for_model(images, model_type):
    if 'inception' in model_type.lower() or 'resnet_v2' in model_type.lower():
        tmp_0 = images[:, :, :, 0] + _R_MEAN
        tmp_1 = images[:, :, :, 1] + _G_MEAN
        tmp_2 = images[:, :, :, 2] + _B_MEAN
        images = tf.stack([tmp_0, tmp_1, tmp_2], 3)
        images = (images / 255.0) * 2.0 - 1.0
        images = tf.image.resize_bilinear(images, [299, 299], align_corners=False)
        return images
    if 'resnet_v1' in model_type.lower() or 'vgg' in model_type.lower():
        images = tf.image.resize_bilinear(images, [224, 224], align_corners=False)
        tmp_0 = images[:, :, :, 0] - _R_MEAN
        tmp_1 = images[:, :, :, 1] - _G_MEAN
        tmp_2 = images[:, :, :, 2] - _B_MEAN
        images = tf.stack([tmp_0, tmp_1, tmp_2], 3)
        return images


def load_images_with_target_label(input_dir, saliency_dir):
    images = []
    saliency_maps = []
    filenames = []
    target_labels_1000 = []
    target_labels_1001 = []
    idx = 0
    dev = pd.read_csv(os.path.join(FLAGS.csv_file))
    filename2label = {dev.iloc[i]['ImageId']: dev.iloc[i]['TargetClass'] for i in range(len(dev))}
    for filename in filename2label.keys():
        image = cv2.imread(os.path.join(input_dir, filename+'.png'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image).astype(np.float)
        images.append(image)

        if not os.path.exists(os.path.join(saliency_dir, filename+'.png')):
            raise NotImplementedError('saliency map not exists!')
        saliency_map = cv2.imread(os.path.join(saliency_dir, filename+'.png'))
        saliency_maps.append(saliency_map)
        filenames.append(filename)
        target_labels_1000.append(filename2label[filename]-1)
        target_labels_1001.append(filename2label[filename])
        idx += 1
        if idx == FLAGS.batch_size:
            images = np.array(images)
            saliency_maps = np.ones(np.array(saliency_maps).shape) - np.array(saliency_maps) / 255.0
            yield filenames, images, target_labels_1000, target_labels_1001, saliency_maps

            filenames = []
            images = []
            saliency_maps = []
            target_labels_1000 = []
            target_labels_1001 = []
            idx = 0
    if idx > 0:
        images = np.array(images)
        saliency_maps = np.ones(np.array(saliency_maps).shape) - np.array(saliency_maps) / 255.0
        yield filenames, images, target_labels_1000, target_labels_1001, saliency_maps


def save_images(images, filenames, output_dir):  # ***
    """
    Saves images to the output directory.
    :param images: array with mini-batch of images
    :param filenames: list of filenames without path. If number of file names in this list less than number of images in
        the mini-batch then only first len(filenames) images will be saved.
    :param output_dir: directory where to save images
    """
    for i, filename in enumerate(filenames):
        image = images[i, :, :, :]
        image[:, :, 0] += _R_MEAN
        image[:, :, 1] += _G_MEAN
        image[:, :, 2] += _B_MEAN
        image = cv2.resize(image, (299, 299))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, filename+'.png'), image)


def check_or_create_dir(directory):
    """Check if directory exists otherwise create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def target_graph(x, y_1, y_2, x_max, x_min, grad, saliency_map, gaussian_kernel, it):
    """
    The computing graph of generating adversarial examples.
    :param x: the image
    :param y_1: the target-label for vgg-models(1000)
    :param y_2: the target-label for inception-models(1001)
    :param x_max: the upper bound of x
    :param x_min: the lower bound of x
    :param grad: the gradient from last iteration
    :param saliency_map: the saliency map of x
    :param gaussian_kernel: the gaussian-kernel
    :param it: the max-iter of each iteration
    :return: adversarial example of this iteration and corresponding gradient
    """
    eps = max_epsilon
    alpha = eps / it

    one_hot_1001 = tf.one_hot(y_2, 1001)
    one_hot_1000 = tf.one_hot(y_1, 1000)

    saliency_x = tf.multiply(x, saliency_map)
    xs = [x, saliency_x]

    # get each model's gradient
    noise_model = []

    for x_hat in xs:

        x_inc = preprocess_for_model(x_hat, 'inception')

        # Resnet_v2_152
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            _, end_points_resnet_v2_152 = resnet_v2.resnet_v2_152(x_inc, reuse=tf.AUTO_REUSE, is_training=False,
                                                                  num_classes=1001, scope='resnet_v2_152')
            end_points_resnet_v2_152['logits'] = tf.squeeze(end_points_resnet_v2_152['resnet_v2_152/logits'], [1, 2])
        logit_resnet = end_points_resnet_v2_152['logits']
        loss_resnet = tf.losses.softmax_cross_entropy(one_hot_1001, logit_resnet / 5, label_smoothing=0.0, weights=1.0)
        resnet_noise = tf.gradients(loss_resnet, x_hat)[0]
        resnet_noise = tf.multiply(resnet_noise, saliency_map)
        noise_model.append(resnet_noise)

        # vgg_19
        with slim.arg_scope(vgg.vgg_arg_scope()):
            logits_vgg_19, _ = vgg.vgg_19(x_hat, num_classes=1000, is_training=False, scope='vgg_19')
        tf.get_variable_scope().reuse_variables()
        logit_vgg = logits_vgg_19
        loss_vgg = tf.losses.softmax_cross_entropy(one_hot_1000, logit_vgg / 5, label_smoothing=0.0, weights=1.0)
        vgg_noise = tf.gradients(loss_vgg, x_hat)[0]
        vgg_noise = tf.multiply(vgg_noise, saliency_map)
        noise_model.append(vgg_noise)

        # inception_v4
        with slim.arg_scope(inception.inception_v4_arg_scope()):
            _, end_points_inc_v4 = inception.inception_v4(x_inc, reuse=tf.AUTO_REUSE, num_classes=1001, is_training=False,
                                                          scope='InceptionV4')
        logit_inc = end_points_inc_v4['AuxLogits']
        loss_inc = tf.losses.softmax_cross_entropy(one_hot_1001, logit_inc / 5, label_smoothing=0.0, weights=1.0)
        inc_noise = tf.gradients(loss_inc, x_hat)[0]
        inc_noise = tf.multiply(inc_noise, saliency_map)
        noise_model.append(inc_noise)

        # inception_resnet
        with slim.arg_scope(inception.inception_resnet_v2_arg_scope()):
            _, end_points_inc_res = inception.inception_resnet_v2(
                x_inc, reuse=tf.AUTO_REUSE, num_classes=1001, is_training=False, scope='InceptionResnetV2')
        logit_inc_res = end_points_inc_res['AuxLogits']
        loss_inc_res = tf.losses.softmax_cross_entropy(one_hot_1001, logit_inc_res / 5, label_smoothing=0.0, weights=1.0)
        inc_res_noise = tf.gradients(loss_inc_res, x_hat)[0]
        inc_res_noise = tf.multiply(inc_res_noise, saliency_map)
        noise_model.append(inc_res_noise)

    # noise fusing and normalization
    noise = noise_fusing(noise_model, batch_size, gaussian_kernel)
    noise = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True)

    noise = momentum * grad + noise

    x = x - alpha * tf.sign(noise)

    x = tf.clip_by_value(x, x_min, x_max)
    return x, noise


def process_transforamation(anglex, angley, imgs, saliency_maps, xmin, xmax):
    """
    Process Perspective projection to images with operator.
    :param anglex: perspective angle_x
    :param angley: perspective angle_y
    :param imgs: x
    :param saliency_maps: saliency_maps
    :param xmin: x_min
    :param xmax: x_max
    :return: Transformed images
    """
    x_max_angle = []
    x_min_angle = []
    original_imgs_angle = []
    saliencymaps_angle = []

    warp = perspective_trandformation(FLAGS.image_width, FLAGS.image_height, anglex, angley)

    for j in range(len(imgs)):
        original_imgs_angle.append(
            cv2.warpPerspective(imgs[j], warp, (FLAGS.image_width, FLAGS.image_width)))
        saliencymaps_angle.append(
            cv2.warpPerspective(saliency_maps[j], warp, (299, 299)))
        x_max_angle.append(
            cv2.warpPerspective(xmax[j], warp, (FLAGS.image_width, FLAGS.image_width)))
        x_min_angle.append(
            cv2.warpPerspective(xmin[j], warp, (FLAGS.image_width, FLAGS.image_width)))

    original_imgs_angle = np.array(original_imgs_angle)
    saliencymaps_angle = np.array(saliencymaps_angle)
    x_min_angle = np.array(x_min_angle)
    x_max_angle = np.array(x_max_angle)

    return original_imgs_angle, saliencymaps_angle, x_min_angle, x_max_angle


def get_gaussian_kernel(radius, num_of_models):
    """
    Calculate Gaussian kernel.
    """
    import scipy.stats as st

    x = np.linspace(-radius, radius, 2 * radius + 1)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    gaussian_for_one_model = kernel_raw / kernel_raw.sum()
    gaussian_for_one_model = np.expand_dims(gaussian_for_one_model, 2)
    gaussian_for_one_model = np.expand_dims(gaussian_for_one_model, 3)
    gaussian_raw = np.repeat(gaussian_for_one_model, num_of_models * 2, axis=2)
    gaussian_kernel = gaussian_raw / (num_of_models * 2)

    return gaussian_kernel


def main(_):
    recent_batch = 0
    start = time.time()
    eps = max_epsilon
    batch_shape = [batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    check_or_create_dir(FLAGS.output_dir)

    with tf.Graph().as_default():

        # setup the placeholders
        s_map_ph = tf.placeholder(tf.float32, shape=[None, 299, 299, 3])  # placeholder for saliency_maps
        raw_input_ph = tf.placeholder(tf.float32, shape=[None, 299, 299, 3])  # placeholder for images
        y_1_ph = tf.constant(np.zeros([batch_size]), tf.int64)  # placeholder for target label-1000
        y_2_ph = tf.constant(np.zeros([batch_size]), tf.int64)  # placeholder for target label-1001
        grad_ph = tf.constant(np.zeros(batch_shape), tf.float32)  # placeholder for gradients
        iter_ph = tf.constant(0, tf.int8)
        # Gaussian kernel used to fuse noises
        gaussian = get_gaussian_kernel(radius=3, num_of_models=4)
        gaussian_kernel = tf.constant(value=gaussian, shape=gaussian.shape, dtype=tf.float32)

        x_input_vgg = preprocess_for_model(raw_input_ph, 'vgg')  # resize images to vgg-style
        x_input_ph = tf.placeholder(tf.float32, shape=batch_shape)
        x_max_tf = x_input_ph + eps
        x_min_tf = x_input_ph - eps
        s_map_vgg = tf.image.resize_images(s_map_ph, [FLAGS.image_height, FLAGS.image_width],
                                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # computing adversarial examples
        x_adv, noise_tf = target_graph(x_input_ph, y_1_ph, y_2_ph, x_max_tf, x_min_tf, grad_ph, s_map_vgg,
                                       gaussian_kernel, iter_ph)

        # load models
        s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV4'))
        s2 = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2'))
        s3 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2_152'))
        s4 = tf.train.Saver(slim.get_model_variables(scope='vgg_19'))

        tf.get_variable_scope().reuse_variables()

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        with tf.compat.v1.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            s1.restore(sess, model_checkpoint_map['inception_v4'])
            s2.restore(sess, model_checkpoint_map['inception_resnet_v2'])
            s3.restore(sess, model_checkpoint_map['resnet_v2_152'])
            s4.restore(sess, model_checkpoint_map['vgg_19'])

            tot_images = 0

            for filenames, raw_images, target_labels_1000, target_labels_1001, s_maps in \
                    load_images_with_target_label(FLAGS.input_dir, FLAGS.saliency_maps):
                tot_images += len(filenames)

                recent_batch += 1
                print('recent batch:', recent_batch)

                if os.path.exists(os.path.join(FLAGS.output_dir, filenames[batch_size - 1]+'.png')):
                    print('file exist!')
                    continue

                original_x = sess.run(x_input_vgg, feed_dict={raw_input_ph: raw_images})
                x_max = sess.run(x_max_tf, feed_dict={x_input_ph: original_x})
                x_min = sess.run(x_min_tf, feed_dict={x_input_ph: original_x})
                branch_noise = np.zeros(shape=batch_shape)

                noise = []
                min_angle = int(-num_angles + 1)
                max_angle = int(num_angles + 1)
                step = 0
                for i in range(min_angle, max_angle):
                    step += 1
                    if i <= 0:
                        anglex = angley = i
                    else:
                        anglex = angley = num_angles - i

                    print('now angle:', anglex)
                    # generate angle images and corresponding boundaries
                    original_x_angle, s_maps_angle, x_min_angle, x_max_angle = process_transforamation(anglex, angley,
                                                                                                        original_x,
                                                                                                        s_maps,
                                                                                                        xmax=x_max,
                                                                                                        xmin=x_min)
                    tau = (eps / num_angles) * step
                    adv_imgs_angle = original_x_angle - tau * np.sign(branch_noise)
                    adv_imgs_angle = np.clip(adv_imgs_angle, x_min_angle, x_max_angle)

                    # compute gradient for each angle
                    for _ in range(0, inner_iters):
                        adv_imgs_angle, branch_noise = sess.run([x_adv, noise_tf],
                                                                feed_dict={x_input_ph: adv_imgs_angle,
                                                                           y_1_ph: target_labels_1000,
                                                                           y_2_ph: target_labels_1001,
                                                                           x_max_tf: x_max_angle,
                                                                           x_min_tf: x_min_angle,
                                                                           grad_ph: branch_noise,
                                                                           s_map_ph: s_maps_angle,
                                                                           iter_ph: inner_iters})
                    if anglex == 0:
                        print('record noise.')
                        noise.append(branch_noise)
                        print('reset noise and step.')
                        step = 0
                        branch_noise = np.zeros(shape=[batch_size, FLAGS.image_width, FLAGS.image_height, 3])

                    if i == max_angle - 1:
                        noise = np.sum(np.array(noise), axis=0)
                        adv_imgs = original_x
                        for _ in range(0, last_iters):
                            adv_imgs, noise = sess.run([x_adv, noise_tf], feed_dict={x_input_ph: adv_imgs,
                                                                                     y_1_ph: target_labels_1000,
                                                                                     y_2_ph: target_labels_1001,
                                                                                     x_max_tf: x_max,
                                                                                     x_min_tf: x_min,
                                                                                     grad_ph: noise,
                                                                                     s_map_ph: s_maps,
                                                                                     iter_ph: last_iters})
                save_images(adv_imgs, filenames, FLAGS.output_dir)

        end = time.time()
        print("total time is {}".format(end - start))


if __name__ == '__main__':
    tf.app.run()
