import argparse
import os
from PIL import Image

parser = argparse.ArgumentParser(description='Make label images more human-readable')
parser.add_argument('-i', '--input_dir', help='Input Directory', required=True)


def _main(args):
    input_dir = os.path.expanduser(args.input_dir)

    if not os.path.exists(input_dir):
        print('ERROR: No such file or diectory: {}'.format(input_dir))
        exit()

    os.makedirs('readable_labels', exist_ok=True)
    palette = get_palette(256)

    for file in os.listdir(input_dir):
        _, ext = os.path.splitext(file)

        if ext in ['.png', '.jpg']:
            print(file)
            label_img = Image.open(os.path.join(input_dir, file))
            label_img = label_img.convert('L')
            label_img.putpalette(palette)
            label_img.save(os.path.join('readable_labels', file))


def get_palette(n):
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


if __name__ == '__main__':
    _main(parser.parse_args())
