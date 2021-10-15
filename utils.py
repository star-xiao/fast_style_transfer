import numpy as np
import tensorflow as tf
import shutil
import imageio


def save_img(out_path, img):
    # 存储图片
    # 转为整形
    img = np.clip(img, 0, 255).astype(np.uint8)
    imageio.imwrite(out_path, img)


def load_img(img_path, max_dim=None, resize=True):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    if resize:
        new_shape = tf.cast([256, 256], tf.int32)
        img = tf.image.resize(img, new_shape)

    # 图像最大边像素点数不超过max_dim
    if max_dim:
        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        if max_dim <= long_dim:
            scale = max_dim / long_dim
            new_shape = tf.cast(shape * scale, tf.int32)
            img = tf.image.resize(img, new_shape)

    # 读入的图片是3维的，此处添加新的维度，使之在之后的计算中维度能够匹配
    img = img[tf.newaxis, :]
    return img


def get_terminal_width():
    width = shutil.get_terminal_size(fallback=(200, 24))[0]
    if width == 0:
        width = 120
    return width
