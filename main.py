# Fast style transform
# VGG-19
# Perceptual losses for real-time style transfer and super-resolution
from train import trainer
import argparse

STYLE_IMAGE = './images/style/wave.jpg'
DATASET_PATH = '/kaggle/input/coco2014/train2014/train2014'
WEIGHTS_PATH = '/kaggle/working/weights/wave/weights_wave'

LEARNING_RATE = 0.001
NUM_EPOCHS = 2
BATCH_SIZE = 16

# fail  A Neural Algorithm of Artistic Style
# CONTENT_WEIGHT = 10000
# STYLE_WEIGHT = 1
# TV_WEIGHT = 1e8

# FastStyle
# CONTENT_WEIGHT = 6e0
# STYLE_WEIGHT = 2e-3
# TV_WEIGHT = 6e2

# fail fast-neural-style-tensorflow
# CONTENT_WEIGHT = 1
# STYLE_WEIGHT = 200
# TV_WEIGHT = 0.001

# fail
# CONTENT_WEIGHT = 1
# STYLE_WEIGHT = 100
# TV_WEIGHT = 0.001

#
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
                        default=WEIGHTS_PATH)
    parser.add_argument('--batch', required=False, type=int,
                        default=BATCH_SIZE)

    args = parser.parse_args()

    parameters = {
        'style_image': args.style,
        'dataset_path': args.dataset,
        'weights_path': args.weights,
        'batch_size': args.batch,
        'content_weight': CONTENT_WEIGHT,
        'style_weight': STYLE_WEIGHT,
        'tv_weight': TV_WEIGHT,
        'learning_rate': LEARNING_RATE,
        'epochs': NUM_EPOCHS,
    }
    trainer(**parameters)


if __name__ == '__main__':
    main()
