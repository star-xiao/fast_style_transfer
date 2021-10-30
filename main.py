# Fast style transform
# VGG-19
# Perceptual losses for real-time style transfer and super-resolution
from train import trainer
import argparse

STYLE_IMAGE = './style/starry night.jpg'
DATASET_PATH = '/kaggle/input/coco2014/train2014/train2014'
LOAD_WEIGHTS_PATH = './weights/starry night/weights_starry night'
SAVE_WEIGHTS_PATH = '/kaggle/working/weights/starry night/weights_starry night'


LEARNING_RATE = 0.0005
NUM_EPOCHS = 1
BATCH_SIZE = 16

CONTENT_WEIGHT = 6
STYLE_WEIGHT = 2e-3
TV_WEIGHT = 6e2


def main():
    parser = argparse.ArgumentParser(
        description='Fast Style Transfer')
    parser.add_argument('--dataset', required=False,
                        default=DATASET_PATH)
    parser.add_argument('--style', required=False,
                        default=STYLE_IMAGE)
    parser.add_argument('--weights', required=False,
                        default=SAVE_WEIGHTS_PATH)
    parser.add_argument('--batch', required=False, type=int,
                        default=BATCH_SIZE)

    args = parser.parse_args()

    parameters = {
        'style_image': args.style,
        'dataset_path': args.dataset,
        'weights_path': args.weights,
        'batch_size': args.batch,
        'load_weights_path': LOAD_WEIGHTS_PATH,
        'content_weight': CONTENT_WEIGHT,
        'style_weight': STYLE_WEIGHT,
        'tv_weight': TV_WEIGHT,
        'learning_rate': LEARNING_RATE,
        'epochs': NUM_EPOCHS,
    }
    trainer(**parameters)


if __name__ == '__main__':
    main()
