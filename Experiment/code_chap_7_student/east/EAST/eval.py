import cv2
import time
import math
import os
import numpy as np
import tensorflow as tf

import locality_aware_nms as nms_locality
import lanms
from tensorflow.python.platform import gfile

tf.app.flags.DEFINE_string('test_data_path', 'pathtodata/east/icdar2015/images/', '')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', 'pathtomodels/east/east_icdar2015_resnet_v1_50_rbox/', '')
tf.app.flags.DEFINE_string('output_dir', './results', '')
tf.app.flags.DEFINE_integer('core_num', 1, '')
tf.app.flags.DEFINE_string('core_version', 'MLU100', '')
tf.app.flags.DEFINE_string('precision', 'float32', '')
tf.app.flags.DEFINE_integer('number', 1, '')
tf.app.flags.DEFINE_integer('batch_size', 1, '')
tf.app.flags.DEFINE_bool('no_write_images', False, 'do not write images')

import model
from icdar import restore_rectangle

FLAGS = tf.app.flags.FLAGS

def get_images():
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    count = 0
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            for ext in exts:
                if count > FLAGS.number-1:
                    return files
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    count += 1
                    break
    print('Find {} images'.format(len(files)))
    return files


def resize_image(im, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    resize_h = max(32, resize_h)
    resize_w = max(32, resize_w)
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def detect(score_map, geo_map, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    #print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    # nms part
    # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)

    if boxes.shape[0] == 0:
        return None

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]


def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    # some check
    if FLAGS.batch_size not in [1, 4, 16]:
        print ("Error! batsh_size should be one of [1, 4, 16]")
        exit(0)
    if FLAGS.core_num not in [1, 4, 16]:
        print ("Error! core_num should be one of [1, 4, 16]")
        exit(0)
    if FLAGS.core_version not in ["MLU100", "MLU270"]:
        print ("Error! core_version should be MLU100 or MLU270")
        exit(0)
    if FLAGS.precision not in ["float32", "int8"]:
        print ("Error! core_version should be float32 or int8")
        exit(0)
    batch_size = FLAGS.batch_size
    # MLU options
    mlu_config = tf.ConfigProto(allow_soft_placement=True,
            inter_op_parallelism_threads=1,
                        intra_op_parallelism_threads=1)
    mlu_config.mlu_options.core_num = FLAGS.core_num
    mlu_config.mlu_options.core_version = FLAGS.core_version
    mlu_config.mlu_options.precision = FLAGS.precision
    #mlu_config.mlu_options.save_offline_model = True
    #mlu_config.mlu_options.offline_model_name = "east_offline"
    mlu_config.log_device_placement = True
    try:
        os.makedirs(FLAGS.output_dir)
    except OSError as e:
        if e.errno != 17:
            raise
    all_time = 0
    net_time = 0
    with tf.get_default_graph().as_default():
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        with tf.Session(config=mlu_config) as sess:

            pb_file = FLAGS.checkpoint_path;
            with gfile.FastGFile(pb_file,'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
            input_images = tf.get_default_graph().get_tensor_by_name("input_images:0")
            f_score = tf.get_default_graph().get_tensor_by_name("feature_fusion/Conv_7/Sigmoid:0")
            f_geometry = tf.get_default_graph().get_tensor_by_name("feature_fusion/concat_3:0")
            concat = tf.get_default_graph().get_tensor_by_name("concat:0")#

            im_fn_list = get_images()
            batch_num = FLAGS.number / batch_size
            for inx in range(int(batch_num)):
                image_list = im_fn_list[inx * batch_size: (inx+1)*batch_size]
                img_tensor = []
                im_batch = []
                for _i, im_fn in enumerate(image_list):
                    im = cv2.imread(str(im_fn))[:, :, ::-1]
                    im_batch.append(im)
                    start_time = time.time()
                    im_resized, (ratio_h, ratio_w) = resize_image(im)
                    im_resized = np.expand_dims(im_resized, 0)
                    img_tensor.append(im_resized)
                im_input = np.concatenate((img_tensor), 0)
                timer = {'net': 0}
                start = time.time()
                score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: im_input})
                timer['net'] = time.time() - start
                print('{} : net {:.0f}ms'.format(str(inx), timer['net']*1000))
                for index in range(int(batch_size)):
                    boxes = detect(score_map=score[index:index+1,:,:,:], geo_map=geometry[index:index+1,:,:,:])

                    if boxes is not None:
                        boxes = boxes[:, :8].reshape((-1, 4, 2))
                        boxes[:, :, 0] /= ratio_w
                        boxes[:, :, 1] /= ratio_h

                    # save to file
                        res_file = os.path.join(
                            FLAGS.output_dir,
                            '{}.txt'.format(
                                os.path.basename(image_list[index]).split('.')[0]))

                        with open(res_file, 'w') as f:
                            for box in boxes:
                                # to avoid submitting errors
                                box = sort_poly(box.astype(np.int32))
                                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                                    continue
                                f.write('{},{},{},{},{},{},{},{}\r\n'.format(
                                    box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1],
                                ))
                                cv2.polylines(im_batch[index][:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=1)
                    if not FLAGS.no_write_images:
                        img_path = os.path.join(FLAGS.output_dir, os.path.basename(image_list[index]))
                        cv2.imwrite(img_path, im_batch[index][:, :, ::-1])
                duration = time.time() - start_time
                #print('[timing] {}'.format(duration))
                if inx > 0:
                    all_time = all_time + duration
                    net_time = net_time + (timer['net'])
    if FLAGS.number > 1:
        print('net fps: %f' %((FLAGS.number-batch_size)/net_time))
        print('end2end fps: %f' %((FLAGS.number-batch_size)/all_time))
    # writer = tf.summary.FileWriter("./log_dir/",tf.get_default_graph())
    # writer.close()

if __name__ == '__main__':
    tf.app.run()
