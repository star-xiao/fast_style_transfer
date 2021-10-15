from tensorflow.keras import models, layers, utils, backend
import numpy as np
import warnings
import os

WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.1/'
                'vgg19_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/'
                       'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')

_IMAGENET_MEAN = None


def vgg19(include_top=True,
          weights='imagenet',
          input_tensor=None,
          input_shape=None,
          pooling=None,
          classes=1000,
          **kwargs):
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(img_input)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = layers.Flatten(name='flatten')(x)
        x = layers.Dense(4096, activation='relu', name='fc1')(x)
        x = layers.Dense(4096, activation='relu', name='fc2')(x)
        x = layers.Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = models.Model(inputs, x, name='vgg19')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            weights_path = utils.get_file(
                'vgg19_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                file_hash='cbe5617147190e668d6c5d5026f83318')
        else:
            weights_path = utils.get_file(
                'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                file_hash='253f8cb515780f3b799900260a226db6')
        model.load_weights(weights_path)
        if backend.backend() == 'theano':
            utils.convert_all_kernels_in_model(model)
    elif weights is not None:
        model.load_weights(weights)

    return model


def preprocess_input(x):
    """Preprocesses a tensor or Numpy array encoding a batch of images.

    # Arguments
        x: Input Numpy or symbolic tensor, 3D or 4D.
            The preprocessed data is written over the input data
            if the data types are compatible. To avoid this
            behaviour, `numpy.copy(x)` can be used.
        data_format: Data format of the image tensor/array.
        mode: One of "caffe", "tf" or "torch".
            - caffe: will convert the images from RGB to BGR,
                then will zero-center each color channel with
                respect to the ImageNet dataset,
                without scaling.
            - tf: will scale pixels between -1 and 1,
                sample-wise.
            - torch: will scale pixels between 0 and 1 and then
                will normalize each channel with respect to the
                ImageNet dataset.

    # Returns
        Preprocessed tensor or Numpy array.

    # Raises
        ValueError: In case of unknown `data_format` argument.
    """
    mean = np.array([123.68, 116.779, 103.939])

    return (x - mean)


def _obtain_input_shape(input_shape,
                        default_size,
                        min_size,
                        require_flatten,
                        weights=None):
    """Internal utility to compute/validate a model's input shape.

    # Arguments
        input_shape: Either None (will return the default network input shape),
            or a user-provided shape to be validated.
        default_size: Default input width/height for the model.
        min_size: Minimum input width/height accepted by the model.
        data_format: Image data format to use.
        require_flatten: Whether the model is expected to
            be linked to a classifier via a Flatten layer.
        weights: One of `None` (random initialization)
            or 'imagenet' (pre-training on ImageNet).
            If weights='imagenet' input channels must be equal to 3.

    # Returns
        An integer shape tuple (may include None entries).

    # Raises
        ValueError: In case of invalid argument values.
    """
    if weights != 'imagenet' and input_shape and len(input_shape) == 3:
        if input_shape[-1] not in {1, 3}:
            warnings.warn(
                'This model usually expects 1 or 3 input channels. '
                'However, it was passed an input_shape with ' +
                str(input_shape[-1]) + ' input channels.')
        default_shape = (default_size, default_size, input_shape[-1])
    else:
        default_shape = (default_size, default_size, 3)

    if weights == 'imagenet' and require_flatten:
        if input_shape is not None:
            if input_shape != default_shape:
                raise ValueError('When setting `include_top=True` '
                                 'and loading `imagenet` weights, '
                                 '`input_shape` should be ' +
                                 str(default_shape) + '.')
        return default_shape

    if input_shape:
        if input_shape is not None:
            if len(input_shape) != 3:
                raise ValueError(
                    '`input_shape` must be a tuple of three integers.')
            if input_shape[-1] != 3 and weights == 'imagenet':
                raise ValueError('The input must have 3 channels; got '
                                 '`input_shape=' + str(input_shape) + '`')
            if ((input_shape[0] is not None and input_shape[0] < min_size) or
                    (input_shape[1] is not None and input_shape[1] < min_size)):
                raise ValueError('Input size must be at least ' +
                                 str(min_size) + 'x' + str(min_size) +
                                 '; got `input_shape=' +
                                 str(input_shape) + '`')
    else:
        if require_flatten:
            input_shape = default_shape
        else:
            input_shape = (None, None, 3)

    if require_flatten:
        if None in input_shape:
            raise ValueError('If `include_top` is True, '
                             'you should specify a static `input_shape`. '
                             'Got `input_shape=' + str(input_shape) + '`')
    return input_shape















# MEAN_PIXEL = np.array([123.68, 116.779, 103.939])
#
#
# def VGGnet(net_path, input_image):
#     LAYERS = (
#         'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
#
#         'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
#
#         'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
#         'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
#
#         'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
#         'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
#
#         'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
#         'relu5_3', 'conv5_4', 'relu5_4'
#     )
#
#     data = scipy.io.loadmat(net_path)
#     # mean = data['normalization'][0][0][0]
#     # mean_pixel = np.mean(mean, axis=(0, 1))
#     # 获取43个层，[0]为去除虚维度
#     layer = data['layers'][0]
#
#     net = {}
#     current = input_image
#     for i, name in enumerate(LAYERS):
#         kind = name[:4]
#         # 卷积层
#         if kind == 'conv':
#             # 获取卷积核权重与编制
#             # [i]：层数(1,1)  1[0]去除虚维度(第一维)(1,)   2[0]:读取当前数据
#             # 3[0]：第一个数据(weights, 包含卷积核与偏置)  4[0]:去除虚维度
#             kernels, bias = layer[i][0][0][0][0]
#             # matconvnet: kernels are [width, height, in_channels, out_channels]
#             # tensorflow: kernels are [height, width, in_channels, out_channels]
#             # 将卷积核转为TensorFlow支持的卷积核
#             kernels = np.transpose(kernels, (1, 0, 2, 3))
#             # 将偏置转为一行
#             bias = bias.reshape(-1)
#             current = conv_layer(current, kernels, bias)
#         # 激活层
#         elif kind == 'relu':
#             current = tf.nn.relu(current)
#         # 池化层
#         elif kind == 'pool':
#             current = pool_layer(current)
#
#
# def conv_layer(input, weights, bias):
#     conv = tf.nn.conv2d(input=input, filters=tf.constant(weights),
#                         strides=(1, 1, 1, 1), padding='SAME')
#     return tf.nn.bias_add(conv, bias)
#
#
# def pool_layer(input):
#     return tf.nn.max_pool2d(input=input, ksize=(1, 2, 2, 1),
#                             strides=(1, 2, 2, 1), padding='SAME')
#
#
# # 图片预处理，去均值，加快训练速度
# def preprocess(image):
#     return image - MEAN_PIXEL
#
#
# def unprocess(image):
#     return image + MEAN_PIXEL











