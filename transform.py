import tensorflow as tf
import os
import time
from trans_net import feed_forward
from utils import load_img, save_img
import argparse

CONTENT_IMAGE = './content/DSC01716.jpg'
WEIGHTS_PATH = './weights/starry night/weights'
start = CONTENT_IMAGE.rindex('/')
OUTPUT_NAME = CONTENT_IMAGE[start+1:]
OUTPUT_NAME = os.path.join('/kaggle/working/', OUTPUT_NAME)


def main():
    parser = argparse.ArgumentParser(
        description='Fast Style Transfer')
    parser.add_argument('--content', required=False,
                        default=CONTENT_IMAGE)
    parser.add_argument('--weights', required=False,
                        default=WEIGHTS_PATH)
    parser.add_argument('--max_dim', required=False, type=int,
                        default=2500)
    parser.add_argument('--output', required=False,
                        default=OUTPUT_NAME)

    args = parser.parse_args()

    parameters = {
        'content_image': args.content,
        'weights': args.weights,
        'max_dim': args.max_dim,
        'output': args.output,
    }

    transfer(**parameters)


def transfer(content_image, weights, max_dim, output):
    network = feed_forward()
    network.load_weights(weights).expect_partial()
    image = load_img(content_image, max_dim=max_dim, resize=False)
    print('Transferring image...')
    t1 = time.time()
    image = network(image)
    image = tf.clip_by_value(image, 0.0, 255.0)
    t2 = time.time()
    print('time:{}'.format(t2 - t1))
    image = image[0]
    save_img(output, image)


if __name__ == '__main__':
    main()
