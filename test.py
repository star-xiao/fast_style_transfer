import tensorflow as tf
from trans_net import feed_forward, tran_conv_2d, resize_conv_2d, conv_2d
from train import StyleContent
from utils import load_img


def main():
    x = tf.random.uniform([4, 256, 256, 3], maxval=256)
    # net = tran_conv_2d(64, 3, 2)
    # net2 = resize_conv_2d(64, 3, 2)
    net3 = conv_2d(3, 9, 1)
    y = net3(x)
    a = 2
    # style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
    #                 'block4_conv1', 'block5_conv1']
    # content_layers = ['block5_conv2']
    # x = tf.random.uniform([4, 256, 256, 3], maxval=256)
    # style_image = './style/The Starry Night.jpg'
    # network = feed_forward()
    # y = network(x)
    # extractor = StyleContent(style_layers, content_layers)
    # style_image = load_img(style_image, resize=False)
    # style_target = extractor(style_image * 255.0)['style']


if __name__ == '__main__':
    main()
