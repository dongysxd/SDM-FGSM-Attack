import os.path
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from cv2 import imread
from nets import inception, vgg, resnet_v2, resnet_v1
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
slim = tf.contrib.slim
tf.flags.DEFINE_string('input_dir', './datasets/images', 'dev_data path')
tf.flags.DEFINE_string('checkpoint_path', './checkpoints/', 'Directory of checkpoint path')
tf.flags.DEFINE_integer('image_width', 299, 'Width of each input images.')
tf.flags.DEFINE_integer('image_height', 299, 'Height of each input images.')
tf.flags.DEFINE_integer('batch_size', 100, 'Number of batch size')
tf.flags.DEFINE_integer('num_classes', 1001, 'Number of Classes')
tf.flags.DEFINE_string('model_type', 'res_v2_152',
                       'vgg_19|inception_v4|inc_res_v2|ens_adv_inception_resnet_v2|ens3_adv_inception_v3|'
                       'ens4_adv_inception_v3|res_v2_152')
tf.flags.DEFINE_string('log', './test-log.txt', 'result')
tf.flags.DEFINE_string('csv_file', './dataset/data.csv', 'csv_file path')
FLAGS = tf.flags.FLAGS
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def preprocess_for_model(images, model_type):
    if 'inc' in model_type.lower() or 'resnet_v2' in model_type.lower():
        images = (images / 255.0) * 2.0 - 1.0
        return images

    if 'resnet_v1' in model_type.lower() or 'vgg' in model_type.lower():
        _R_MEAN = 123.68
        _G_MEAN = 116.78
        _B_MEAN = 103.94
        images = tf.image.resize_bilinear(images, [224, 224], align_corners=False)
        tmp_0 = images[:, :, :, 0] - _R_MEAN
        tmp_1 = images[:, :, :, 1] - _G_MEAN
        tmp_2 = images[:, :, :, 2] - _B_MEAN
        images = tf.stack([tmp_0, tmp_1, tmp_2], 3)
        return images


def load_images_with_target_label(input_dir):
    images = []
    filenames = []
    target_labels = []
    idx = 0

    dev = pd.read_csv(FLAGS.csv_file)
    filename2label = {dev.iloc[i]['ImageId']: dev.iloc[i]['TargetClass'] for i in range(len(dev))}
    for filename in filename2label.keys():
        if not os.path.exists(os.path.join(input_dir, filename+'.png')):
            continue
        image = imread(os.path.join(input_dir, filename+'.png'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
        filenames.append(filename)
        if 'inc' in FLAGS.model_type or 'res_v2' in FLAGS.model_type:
            target_labels.append(filename2label[filename])
        else:
            target_labels.append(filename2label[filename]-1)
        idx += 1
        if idx % FLAGS.batch_size == 0:
            yield filenames, np.array(images), target_labels
            images = []
            filenames = []
            target_labels = []


def main(_):
    if FLAGS.model_type == 'ens3_adv_inception_v3':
        FLAGS.num_classes = 1001
        arg_scope = inception.inception_v3_arg_scope()
        checkpoint_file = os.path.join(FLAGS.checkpoint_path, 'ens3_adv_inception_v3.ckpt')
    elif FLAGS.model_type == 'inception_v3':
        FLAGS.num_classes = 1001
        arg_scope = inception.inception_v3_arg_scope()
        checkpoint_file = os.path.join(FLAGS.checkpoint_path, 'inception_v3.ckpt')
    elif FLAGS.model_type == 'ens4_adv_inception_v3':
        FLAGS.num_classes = 1001
        arg_scope = inception.inception_v3_arg_scope()
        checkpoint_file = os.path.join(FLAGS.checkpoint_path, 'ens4_adv_inception_v3.ckpt')
    elif FLAGS.model_type == 'inception_v4':
        FLAGS.num_classes = 1001
        arg_scope = inception.inception_v4_arg_scope()
        checkpoint_file = os.path.join(FLAGS.checkpoint_path, 'inception_v4.ckpt')
    elif FLAGS.model_type == 'vgg_19':
        FLAGS.num_classes = 1000
        arg_scope = vgg.vgg_arg_scope()
        checkpoint_file = os.path.join(FLAGS.checkpoint_path, 'vgg_19.ckpt')
    elif FLAGS.model_type == 'vgg_16':
        FLAGS.num_classes = 1000
        arg_scope = vgg.vgg_arg_scope()
        checkpoint_file = os.path.join(FLAGS.checkpoint_path, 'vgg_16.ckpt')
    elif FLAGS.model_type == 'inc_res_v2':
        FLAGS.num_classes = 1001
        arg_scope = inception.inception_resnet_v2_arg_scope()
        checkpoint_file = os.path.join(FLAGS.checkpoint_path, 'inception_resnet_v2.ckpt')
    elif FLAGS.model_type == 'ens_adv_inception_resnet_v2':
        FLAGS.num_classes = 1001
        arg_scope = inception.inception_resnet_v2_arg_scope()
        checkpoint_file = os.path.join(FLAGS.checkpoint_path, 'ens_adv_inception_resnet_v2.ckpt')
    elif FLAGS.model_type == 'res_v2_152':
        FLAGS.num_classes = 1001
        arg_scope = resnet_v2.resnet_arg_scope()
        checkpoint_file = os.path.join(FLAGS.checkpoint_path, 'resnet_v2_152/resnet_v2_152.ckpt')
    elif FLAGS.model_type == 'resnet_v1_50':
        FLAGS.num_classes = 1000
        arg_scope = resnet_v1.resnet_arg_scope()
        checkpoint_file = os.path.join(FLAGS.checkpoint_path, 'resnet_v1_50.ckpt')
    elif FLAGS.model_type == 'resnet_v1_152':
        FLAGS.num_classes = 1000
        arg_scope = resnet_v1.resnet_arg_scope()
        checkpoint_file = os.path.join(FLAGS.checkpoint_path, 'resnet_v1_152.ckpt')
    else:
        raise NotImplementedError('model {} does not exist.'.format(FLAGS.model_type))

    with slim.arg_scope(arg_scope):

        input_images = tf.placeholder(dtype=tf.float32, shape=[None, 299, 299, 3])
        input_labels = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.num_classes])
        is_training = tf.placeholder(dtype=tf.bool)

        if FLAGS.model_type == 'ens3_adv_inception_v3':
            processed_imgs = preprocess_for_model(input_images, 'inception_v3')
            _, end_points = inception.inception_v3(
                processed_imgs, num_classes=FLAGS.num_classes, reuse=tf.AUTO_REUSE, is_training=False)
            logits = end_points['AuxLogits']

        elif FLAGS.model_type == 'ens4_adv_inception_v3':
            processed_imgs = preprocess_for_model(input_images, 'inception_v3')
            _, end_points = inception.inception_v3(
                processed_imgs, num_classes=FLAGS.num_classes, reuse=tf.AUTO_REUSE, is_training=False)
            logits = end_points['AuxLogits']

        elif FLAGS.model_type == 'inception_v4':
            processed_imgs = preprocess_for_model(input_images, 'inception_v4')
            _, end_points = inception.inception_v4(
                processed_imgs, num_classes=FLAGS.num_classes, reuse=tf.AUTO_REUSE, is_training=False)
            logits = end_points['AuxLogits']

        elif FLAGS.model_type == 'vgg_19':
            processed_imgs = preprocess_for_model(input_images, 'vgg_19')
            logits_vgg_19, end_points_vgg_19 = vgg.vgg_19(processed_imgs, is_training=is_training,
                                                          num_classes=FLAGS.num_classes)
            logits = end_points_vgg_19['vgg_19/fc8']

        elif FLAGS.model_type == 'res_v2_152':
            processed_imgs = preprocess_for_model(input_images, 'resnet_v2_152')
            logits_resnet_v2_152, end_points_resnet_v2_152 = resnet_v2.resnet_v2_152(processed_imgs,
                                                                                     is_training=is_training,
                                                                                     num_classes=FLAGS.num_classes)
            logits = tf.squeeze(end_points_resnet_v2_152['resnet_v2_152/logits'], [1, 2])

        elif FLAGS.model_type == 'inc_res_v2':
            processed_imgs = preprocess_for_model(input_images, 'inception_v3')
            _, end_points_inception_resnet = inception.inception_resnet_v2(processed_imgs, is_training=is_training,
                                                                           num_classes=FLAGS.num_classes)
            logits = end_points_inception_resnet['AuxLogits']

        elif FLAGS.model_type == 'ens_adv_inception_resnet_v2':
            processed_imgs = preprocess_for_model(input_images, 'inception_v3')
            _, end_points_inception_resnet = inception.inception_resnet_v2(processed_imgs, is_training=is_training,
                                                                           num_classes=FLAGS.num_classes)
            logits = end_points_inception_resnet['AuxLogits']

        else:
            raise NotImplementedError('Please select models from vgg_19|inception_v4|inc_res_v2|ens_adv_inception_resnet_v2|ens3_adv_inception_v3|ens4_adv_inception_v3|res_v2_152.')

        params = slim.get_variables_to_restore()
        restorer = tf.train.Saver(params)
        pred = tf.argmax(logits, axis=1)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=input_labels))
        correct = tf.equal(pred, tf.argmax(input_labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            restorer.restore(sess, checkpoint_file)
            print('test_dataset:', FLAGS.input_dir)
            print('test attack success rate from {}'.format(FLAGS.model_type))
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            accuracy_all = 0
            count = 0
            tot_images = 0

            for _, test_images, test_labels in load_images_with_target_label(FLAGS.input_dir):
                tot_images += len(test_images)

                test_labels = slim.one_hot_encoding(test_labels, FLAGS.num_classes).eval()
                cost_values, accuracy_value = sess.run([cost, accuracy],
                                                       feed_dict={input_images: test_images, input_labels: test_labels,
                                                                  is_training: False})
                print(accuracy_value)
                accuracy_all += accuracy_value
                count += 1
            print("result:", accuracy_all / count)
            with open(FLAGS.log, 'a+') as f:
                f.write('Experiment:')
                f.write(FLAGS.input_dir + ' ' + FLAGS.model_type + ':')
                f.write(str(accuracy_all / count))
                f.write('\n')
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    tf.app.run()
