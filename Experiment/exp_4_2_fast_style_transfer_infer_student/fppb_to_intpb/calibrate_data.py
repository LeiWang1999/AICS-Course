import cv2
import numpy as np
import tensorflow as tf

def read_image_batch(file_list, batch_size, iters):
  if batch_size > len(file_list):
    raise ValueError("batch_size must be less equal than file_list size.")
  batch_data = []
  start = iters * batch_size
  if start < len(file_list):
    end = start + batch_size if (start + batch_size) < len(file_list) else len(file_list)
    for path in file_list[start : end]:
      batch_data.append(cv2.imread(path))
  else:
    print("Warning: batch_size * num_runs > file_list size.")
    for path in file_list[-batch_size : ]:
      batch_data.append(cv2.imread(path))
  return batch_data

def read_file(input_image_list):
  with open(input_image_list, "r") as f:
    images_path = [line.strip() for line in f.read().splitlines()]
  return images_path

class efficientNet_preprocess(object):
  def __init__(self, params):
    self.iter = -1
    self.file_list = read_file(params['data_path'])
    self.batch_size = int(params['batch_size'])
    self.means = params['mean'].replace(" ","").split(",")
    self.means = [ float(mean) for mean in self.means ]
    crop = params['crop'].replace(" ","").split(",")
    crop = [ int(c) for c in crop ]
    self.out_h, self.out_w = crop
    self.input_tensor_names = params['input_tensor_names']
    self.color_mode = params['color_mode']
    self.std = params['std'].replace(" ","").split(",")
    self.std = [ float(std) for std in self.std ]
    if self.color_mode == "rgb":
      self.means = self.means[::-1]

  def next(self):
    self.iter += 1
    batch_data = self.efficientNet_read_image(self.file_list, self.batch_size, self.iter)
    batch_data = [self.efficientNet_preprocess_image(image) for image in batch_data]
    batch_data = np.asarray(batch_data)
    means = np.array(self.means)
    batch_data = np.reshape(np.asarray(batch_data, np.float32),
            [self.batch_size, self.out_h, self.out_w, 3])
    batch_data = (batch_data - means) / self.std
    if len(batch_data.shape) == 1:
      raise ValueError("Preprocessed images must have the same shape,"
                       " if batch_size is greater than 1.")
    return {self.input_tensor_names[0] : batch_data}

  def efficientNet_read_image(self, file_list, batch_size, iters):
    if batch_size > len(file_list):
      raise ValueError("batch_size must be less equal than file_list size.")
    batch_data = []
    start = iters * batch_size
    if start < len(file_list):
      end = start + batch_size if (start + batch_size) < len(file_list) else len(file_list)
      for path in file_list[start : end]:
        batch_data.append(path)
    else:
      print("Warning: batch_size * num_runs > file_list size.")
      for path in file_list[-batch_size : ]:
        batch_data.append(path)
    return batch_data


  def _decode_and_center_crop(self, image_bytes,image_size):
    shape = tf.image.extract_jpeg_shape(image_bytes)
    image_height = shape[0]
    image_width = shape[1]
    padded_center_crop_size = tf.cast(
            ( 0.875 *
                tf.cast(tf.minimum(image_height, image_width), tf.float32)),
            tf.int32)
    offset_height = ((image_height - padded_center_crop_size) + 1) // 2
    offset_width = ((image_width - padded_center_crop_size) + 1) // 2
    crop_window = tf.stack([offset_height, offset_width,
        padded_center_crop_size, padded_center_crop_size])
    image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
    image = tf.image.resize_bicubic([image], [image_size, image_size])[0]
    return image

  def efficientNet_preprocess_image(self, image):
    filenames = tf.constant(image)
    image_string = tf.read_file(filenames)
    image =self._decode_and_center_crop(image_string, self.out_h)
    image = tf.reshape(image, [self.out_h, self.out_w, 3])
    image = tf.image.convert_image_dtype(image, tf.float32)
    sess = tf.Session()
    return (sess.run(image))


class no_preprocess_cali(object):
  """Do nothing."""
  def __init__(self, params):
    self._iter = -1
    self.file_list = read_file(params['data_path'])
    self.batch_size = int(params['batch_size'])
    self.input_tensor_names = params['input_tensor_names']

  def next(self):
    self._iter += 1
    batch_data = read_image_batch(self.file_list, self.batch_size, self._iter)
    batch_data = np.asarray(batch_data)
    if len(batch_data.shape) == 1:
      raise ValueError("Preprocessed images must have the same shape,"
                       " if batch_size is greater than 1.")
    return {self.input_tensor_names[0] : batch_data}


class default_preprocess_cali(object):
  def __init__(self, params):
    self.iter = -1
    self.file_list = read_file(params['data_path'])
    self.batch_size = int(params['batch_size'])
    self.means = params['mean'].replace(" ","").split(",")
    self.means = [ float(mean) for mean in self.means ]
    self.std = params['std'].replace(" ","").split(",")
    self.std = [ float(std) for std in self.std ]
    crop = params['crop'].replace(" ","").split(",")
    crop = [ int(c) for c in crop ]
    self.out_h, self.out_w = crop
    self.color_mode = params['color_mode']
    self.input_tensor_names = params['input_tensor_names']
    if self.color_mode == "rgb":
      self.means = self.means[::-1]

  def next(self):
    self.iter += 1
    batch_data = read_image_batch(self.file_list, self.batch_size, self.iter)
    batch_data = [self.default_preprocess(image) for image in batch_data]
    batch_data = np.asarray(batch_data)
    if len(batch_data.shape) == 1:
      raise ValueError("Preprocessed images must have the same shape,"
                       " if batch_size is greater than 1.")
    return {self.input_tensor_names[0] : batch_data}

  def default_preprocess(self, image):
    if self.color_mode == 'rgb':
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (self.out_w, self.out_h))
    image = (image - self.means) / self.std
    return image


class slim_vgg_preprocess_cali(object):
  def __init__(self, params):
    self.iter = -1
    self.file_list = read_file(params['data_path'])
    self.batch_size = int(params['batch_size'])
    crop = params['crop'].replace(" ","").split(",")
    crop = [ int(c) for c in crop ]
    self.out_h, self.out_w = crop
    self.input_tensor_names = params['input_tensor_names']

  def next(self):
    self.iter += 1
    batch_data = read_image_batch(self.file_list, self.batch_size, self.iter)
    batch_data = [self.slim_vgg_preprocess(image) for image in batch_data]
    batch_data = np.asarray(batch_data)
    if len(batch_data.shape) == 1:
      raise ValueError("Preprocessed images must have the same shape,"
                       " if batch_size is greater than 1.")
    return {self.input_tensor_names[0] : batch_data}

  def slim_vgg_preprocess(self, image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mean_r, mean_g, mean_b = 123.68, 116.78, 103.94
    resize_side_min = 256.0
    if resize_side_min < min(self.out_h, self.out_w):
        raise ValueError("vgg_preprocess resize_side_min must be greater equal than out_side_min,"
                         " {} vs. {}".format(resize_side_min, min(self.out_h, self.out_w)))
    image_h = image.shape[0]
    image_w = image.shape[1]

    # resize_bilinear
    scale = resize_side_min / min(image_h, image_w)
    resize_h = int(round(image_h * scale))
    resize_w = int(round(image_w * scale))
    resized_image = cv2.resize(image.astype(np.float32), (resize_w, resize_h))

    # central_crop
    offset_h = (resize_h - self.out_h) / 2
    offset_w = (resize_w - self.out_w) / 2
    croped_image = resized_image[offset_h:offset_h+self.out_h, offset_w:offset_w+self.out_w, :]
    last_image = croped_image - [mean_r, mean_g, mean_b]
    return last_image


class slim_inception_preprocess_cali(object):
  def __init__(self, params):
    self.iter = -1
    self.file_list = read_file(params['data_path'])
    self.batch_size = int(params['batch_size'])
    crop = params['crop'].replace(" ","").split(",")
    crop = [ int(c) for c in crop ]
    self.out_h, self.out_w = crop
    self.input_tensor_names = params['input_tensor_names']

  def next(self):
    self.iter += 1
    batch_data = read_image_batch(self.file_list, self.batch_size, self.iter)
    batch_data = [self.slim_inception_preprocess(image) for image in batch_data]
    batch_data = np.asarray(batch_data)
    if len(batch_data.shape) == 1:
      raise ValueError("Preprocessed images must have the same shape,"
                       " if batch_size is greater than 1.")
    return {self.input_tensor_names[0] : batch_data}

  def slim_inception_preprocess(self, image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_h = image.shape[0]
    image_w = image.shape[1]
    image_norm = image / 255.0

    # central_crop
    offset_h = int((image_h - image_h * 0.875) / 2)
    offset_w = int((image_w - image_w * 0.875) / 2)
    size_h = image_h - offset_h * 2
    size_w = image_w - offset_w * 2
    croped_image = image_norm[offset_h:offset_h+size_h, offset_w:offset_w+size_w, :]

    # resize_bilinear
    resized_image = cv2.resize(croped_image, (self.out_h, self.out_w))
    last_image = (resized_image - 0.5) * 2.0
    return last_image


class yolov3_preprocess_cali(object):
  def __init__(self, params):
    self.iter = -1
    self.file_list = read_file(params['data_path'])
    self.batch_size = int(params['batch_size'])
    crop = params['crop'].replace(" ","").split(",")
    crop = [ int(c) for c in crop ]
    self.out_h, self.out_w = crop
    self.input_tensor_names = params['input_tensor_names']

  def next(self):
    self.iter += 1
    batch_data = read_image_batch(self.file_list, self.batch_size, self.iter)
    batch_data = [self.yolov3_preprocess(image) for image in batch_data]
    batch_data = np.asarray(batch_data)
    if len(batch_data.shape) == 1:
      raise ValueError("Preprocessed images must have the same shape,"
                       " if batch_size is greater than 1.")
    return {self.input_tensor_names[0] : batch_data}

  def yolov3_preprocess(self, image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _  = image.shape
    scale = min(float(self.out_w)/w, float(self.out_h)/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))
    image_paded = np.full(shape=[self.out_h, self.out_w, 3], fill_value=128.0)
    dw, dh = (self.out_w - nw) // 2, (self.out_h-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.
    return image_paded


class yolov2_preprocess_cali(object):
  def __init__(self, params):
    self.iter = -1
    self.file_list = read_file(params['data_path'])
    self.batch_size = int(params['batch_size'])
    crop = params['crop'].replace(" ","").split(",")
    crop = [ int(c) for c in crop ]
    self.out_h, self.out_w = crop
    self.input_tensor_names = params['input_tensor_names']

  def next(self):
    self.iter += 1
    batch_data = read_image_batch(self.file_list, self.batch_size, self.iter)
    batch_data = [self.yolov2_preprocess(image) for image in batch_data]
    batch_data = np.asarray(batch_data)
    if len(batch_data.shape) == 1:
      raise ValueError("Preprocessed images must have the same shape,"
                       " if batch_size is greater than 1.")
    return {self.input_tensor_names[0] : batch_data}

  def yolov2_preprocess(self, image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_h, image_w, _ = image.shape
    image = cv2.resize(image, (self.out_h, self.out_w))
    image = image / 255.0 * 2.0 - 1.0
    return image


class pixellink_preprocess_cali(object):
  def __init__(self, params):
    self.iter = -1
    self.file_list = read_file(params['data_path'])
    self.batch_size = int(params['batch_size'])
    crop = params['crop'].replace(" ","").split(",")
    crop = [ int(c) for c in crop ]
    self.out_h, self.out_w = crop
    self.input_tensor_names = params['input_tensor_names']

  def next(self):
    self.iter += 1
    batch_data = read_image_batch(self.file_list, 1, self.iter)
    batch_data = [self.pixellink_preprocess(image) for image in batch_data]
    batch_data = np.asarray(batch_data[0])
    if len(batch_data.shape) == 1:
      raise ValueError("Preprocessed images must have the same shape,"
                       " if batch_size is greater than 1.")
    return {self.input_tensor_names[0] : batch_data}

  def pixellink_preprocess(self, image):
    imgae = image.astype(np.int32)
    return image


class east_preprocess_cali(object):
  def __init__(self, params):
    self.iter = -1
    self.file_list = read_file(params['data_path'])
    self.batch_size = int(params['batch_size'])
    crop = params['crop'].replace(" ","").split(",")
    crop = [ int(c) for c in crop ]
    self.out_h, self.out_w = crop
    self.input_tensor_names = params['input_tensor_names']

  def next(self):
    self.iter += 1
    batch_data = read_image_batch(self.file_list, self.batch_size, self.iter)
    batch_data = [self.east_preprocess(image) for image in batch_data]
    batch_data = np.asarray(batch_data)
    if len(batch_data.shape) == 1:
      raise ValueError("Preprocessed images must have the same shape,"
                       " if batch_size is greater than 1.")
    return {self.input_tensor_names[0] : batch_data}

  def east_preprocess(self, image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image
    max_side_len=2400
    h, w, _ = image.shape
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
    image = cv2.resize(image, (int(resize_w), int(resize_h)))
    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)
    return image


def get_calibrate_data(params, name="default_preprocess_cali"):
  print("calibrate_data: ", name)
  calibrate_data_map = {
       "no_preprocess_cali": no_preprocess_cali,
       "default_preprocess_cali": default_preprocess_cali,
       "vgg_preprocess_cali": slim_vgg_preprocess_cali,
       "inception_preprocess_cali": slim_inception_preprocess_cali,
       "yolov3_preprocess_cali": yolov3_preprocess_cali,
       "yolov2_preprocess_cali": yolov2_preprocess_cali,
       "east_preprocess_cali": east_preprocess_cali,
       "pixellink_preprocess_cali": pixellink_preprocess_cali,
       "efficientNet_preprocess" : efficientNet_preprocess,
      }
  return calibrate_data_map[name](params)
