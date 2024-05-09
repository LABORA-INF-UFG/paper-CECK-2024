import numpy as np
import os
import glob
from PIL import Image
import sys
import argparse
from efficientnet.keras import center_crop_and_resize


def resize_image(path_image):
    image = Image.open(path_image)
    resized_image_np = center_crop_and_resize(np.array(image), 450)
    resized_image_np = resized_image_np.astype(np.uint8)
    resized_image = Image.fromarray(resized_image_np)
    return resized_image



def main(args):
    redimensionar_imagens(args.dir_images_input)

def redimensionar_imagens(dir_images_input):
    for (dirpath, dirnames, filenames) in os.walk(dir_images_input):
        for filename in filenames:
            abs_filename = os.path.abspath(os.path.join(dirpath, filename))
            dir_name = os.path.basename(os.path.dirname(abs_filename))
            new_dir_name = dir_name+'_450'
            resized_image = resize_image(abs_filename)
            abs_filename = abs_filename.replace(dir_name, new_dir_name)
            if not os.path.exists(os.path.dirname(abs_filename)):
                os.makedirs(os.path.dirname(abs_filename))
            resized_image.save(abs_filename, "JPEG", quality=30, dpi=(300, 300))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_images_input', type=str)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))