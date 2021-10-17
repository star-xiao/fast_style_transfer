import tensorflow as tf
import numpy as np
import time
from tensorflow.keras import optimizers, models, applications
from utils import load_img
from trans_net import feed_forward


def get_vgg_layers(layer_names):
    # 预训练的VGG19网络，不包括最后全连接层, imagenet权重参数
    vgg = applications.VGG19(include_top=False, weights='imagenet')
    # 网络参数不可训练
    vgg.trainable = False
    # 获取各层输出的信息
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model


def preprocess_input(x):
    mean = np.array([123.68, 116.779, 103.939])
    return x - mean


def calculate_gram_matrix(inputs, normalize=True):
    batch_size, height, width, filters = inputs.shape
    inputs = tf.reshape(inputs, (batch_size, height * width, filters))

    tran_f = tf.transpose(inputs, perm=[0, 2, 1])
    gram = tf.matmul(tran_f, inputs)
    if normalize:
        gram /= tf.cast(height * width, tf.float32)

    return gram


def style_loss(style_outputs, style_target):
    # 均方误差MSE
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_target[name]) ** 2)
                           for name in style_outputs.keys()])

    return style_loss / len(style_outputs)


def content_loss(content_outputs, content_target):
    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_target[name]) ** 2)
                             for name in content_outputs.keys()])

    return content_loss / len(content_outputs)


def total_variation_loss(img):
    # 计算两个像素点之间的像素差值，并返回
    x_var = img[:, :, 1:, :] - img[:, :, :-1, :]
    y_var = img[:, 1:, :, :] - img[:, :-1, :, :]
    return tf.reduce_mean(tf.square(x_var)) + tf.reduce_mean(tf.square(y_var))


class StyleContent(models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContent, self).__init__()
        self.vgg = get_vgg_layers(style_layers + content_layers)
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.num_content_layers = len(content_layers)
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        preprocessed_input = preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])
        # 通过Gram矩阵计算风格损失
        style_outputs = [calculate_gram_matrix(style_output) for style_output in style_outputs]
        # 创建内容字典
        content = {
            content_name: value
            for content_name, value in zip(self.content_layers, content_outputs)
        }
        # 创建风格字典
        style = {
            style_name: value
            for style_name, value in zip(self.style_layers, style_outputs)
        }
        return {'content': content, 'style': style}


def trainer(style_image, dataset_path, weights_path, load_weights_path, content_weight, style_weight,
            tv_weight, learning_rate, batch_size, epochs):
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']

    content_layers = ['block4_conv2']

    network = feed_forward()
    # 若中断训练，可重新加载权重
    # network.load_weights(load_weights_path).expect_partial()
    extractor = StyleContent(style_layers, content_layers)
    style_image = load_img(style_image)
    style_target = extractor(style_image * 255.0)['style']
    # 初始化目标内容图
    batch_shape = (batch_size, 256, 256, 3)
    X_batch = np.zeros(batch_shape, dtype=np.float32)
    optimizer = optimizers.Adam(learning_rate=learning_rate)

    # 计算给定值的（加权）平均值
    loss_metric = tf.keras.metrics.Mean()
    sloss_metric = tf.keras.metrics.Mean()
    closs_metric = tf.keras.metrics.Mean()
    tloss_metric = tf.keras.metrics.Mean()

    def train_step(X_batch):
        with tf.GradientTape() as tape:
            content_target = extractor(X_batch * 255.0)['content']
            image = network(X_batch)
            outputs = extractor(image)

            s_loss = style_weight * style_loss(outputs['style'], style_target)
            c_loss = content_weight * content_loss(outputs['content'], content_target)
            t_loss = tv_weight * total_variation_loss(image)
            loss = s_loss + c_loss + t_loss

        grad = tape.gradient(loss, network.trainable_variables)
        optimizer.apply_gradients(zip(grad, network.trainable_variables))

        loss_metric(loss)
        sloss_metric(s_loss)
        closs_metric(c_loss)
        tloss_metric(t_loss)

    # 高性能数据管道
    train_dataset = tf.data.Dataset.list_files(dataset_path + '/*.jpg')
    # 并行读取图像
    train_dataset = train_dataset.map(load_img,
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.shuffle(1024)
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    # GPU 预加载
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    start = time.time()

    for e in range(epochs):
        step = 0
        debug_step = 100
        save_step = 1000

        for img in train_dataset:
            for j, img_p in enumerate(img):
                X_batch[j] = img_p

            step += 1
            train_step(X_batch)
            now = time.time()

            if step % debug_step == 0:
                print('step %s: loss = %s' % (step, loss_metric.result()))
                print('s_loss={}, c_loss={}, t_loss={}'.format(sloss_metric.result(), closs_metric.result(),
                                                               tloss_metric.result()))

            if step % save_step == 0:
                # Save checkpoints
                network.save_weights(weights_path, save_format='tf')
                print('\n=====================================')
                print('       Epoch {} iteration {:.1f}       '.format(e + 1, step))
                print('           Weights saved!              '.format(step))
                print('           time: {:.0f}min{:.0f}s      '.format(((now - start) - (now - start) % 60) / 60, (now - start) % 60))
                print('=====================================\n')

    end = time.time()
    print('Total time: {:.0f}min{:.0f}s '.format(((end - start) - (end - start) % 60) / 60, (end - start) % 60))

    network.save_weights(weights_path, save_format='tf')
    print('\n=====================================')
    print('              All saved!                ')
    print('=====================================\n')
