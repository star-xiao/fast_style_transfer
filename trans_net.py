import tensorflow as tf
from tensorflow.keras import layers, models


class instance_norm(layers.Layer):
    def __init__(self, epsilon=1e-3):
        super(instance_norm, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        # self.gamma = self.add_weight(shape=input_shape[-1],
        #                              initializer='ones',
        #                              trainable=True)
        # self.beta = self.add_weight(shape=input_shape[-1],
        #                             initializer='zeros',
        #                             trainable=True)
        self.beta = tf.Variable(tf.zeros([input_shape[3]]))
        self.gamma = tf.Variable(tf.ones([input_shape[3]]))

    def call(self, inputs):
        # 计算高和宽的均值与方差
        mean, var = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        x = tf.divide(tf.subtract(inputs, mean), tf.sqrt(tf.add(var, self.epsilon)))
        return self.gamma * x + self.beta


class conv_2d(layers.Layer):
    def __init__(self, filters, kernel, stride):
        super(conv_2d, self).__init__()
        pad = kernel // 2  # 取整数,返回商的整数部分 9 // 2 = 4
        self.padding = tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]])
        self.conv2d = layers.Conv2D(filters, kernel, stride, use_bias=False, padding='valid')
        self.instance_norm = instance_norm()

    def call(self, inputs, relu=True):
        # mode = "REFLECT"  映射填充
        x = tf.pad(inputs, self.padding, mode='REFLECT')
        # padding='valid' 时，输出大小等于输入大小减去滤波器大小加上1，最后再除以步长
        # 如输入为(1, 32, 32, 3),经过conv_2d(32, 9, 1)后为 (（32+8）-9+1)/1=32
        x = self.conv2d(x)
        x = self.instance_norm(x)

        if relu:
            x = tf.nn.relu(x)
        return x


class resize_conv_2d(layers.Layer):
    def __init__(self, filters, kernel, stride):
        super(resize_conv_2d, self).__init__()
        self.conv = conv_2d(filters, kernel, stride)
        self.instance_norm = instance_norm()
        self.stride = stride

    def call(self, inputs):
        new_h = inputs.shape[1] * self.stride * 2
        new_w = inputs.shape[2] * self.stride * 2
        # 最近邻插值调整inputs为[new_h, new_w]
        x = tf.image.resize(inputs, [new_h, new_w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        x = self.conv(x)
        x = self.instance_norm(x)
        return tf.nn.relu(x)


class tran_conv_2d(layers.Layer):
    def __init__(self, filters, kernel, stride):
        super(tran_conv_2d, self).__init__()
        self.tran_conv = tf.keras.layers.Conv2DTranspose(filters, kernel, stride, padding='same')
        self.instance_norm = instance_norm()

    def call(self, inputs):
        x = self.tran_conv(inputs)
        x = self.instance_norm(x)

        return tf.nn.relu(x)


class residual(layers.Layer):
    def __init__(self, filters, kernel, stride):
        super(residual, self).__init__()
        self.conv1 = conv_2d(filters, kernel, stride)
        self.conv2 = conv_2d(filters, kernel, stride)

    def call(self, inputs):
        # 步长为1，不需要进行维度匹配
        x = self.conv1(inputs)
        return inputs + self.conv2(x, relu=False)


class feed_forward(models.Model):
    def __init__(self):
        super(feed_forward, self).__init__()
        self.conv1 = conv_2d(32, 9, 1)
        self.conv2 = conv_2d(64, 3, 2)
        self.conv3 = conv_2d(128, 3, 2)
        self.resid1 = residual(128, 3, 1)
        self.resid2 = residual(128, 3, 1)
        self.resid3 = residual(128, 3, 1)
        self.resid4 = residual(128, 3, 1)
        self.resid5 = residual(128, 3, 1)
        # self.tran_conv1 = tran_conv_2d(64, 3, 2)
        # self.tran_conv2 = tran_conv_2d(32, 3, 2)
        self.resize_conv1 = resize_conv_2d(64, 3, 2)
        self.resize_conv2 = resize_conv_2d(32, 3, 2)
        self.conv4 = conv_2d(3, 9, 1)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.resid1(x)
        x = self.resid2(x)
        x = self.resid3(x)
        x = self.resid4(x)
        x = self.resid5(x)
        x = self.resize_conv1(x)
        x = self.resize_conv2(x)
        x = self.conv4(x, relu=False)
        return tf.nn.tanh(x) * 150 + 255. / 2
