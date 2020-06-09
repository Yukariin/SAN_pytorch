import argparse
import glob
import os
import io
import sqlite3

from PIL import Image
import numpy as np
import skimage
import skimage.io
import skimage.transform
from tqdm import tqdm


def imread(path):
    img = skimage.io.imread(path)
    if img.ndim != 3:
        img = skimage.color.gray2rgb(img)
    if img.shape[-1] == 4:
        img = img[..., :3]

    return img


def gradients(x):
    return np.mean(((x[:-1, :-1, :] - x[1:, :-1, :]) ** 2 + (x[:-1, :-1, :] - x[:-1, 1:, :]) ** 2))


def convert(np_img):
    img_byte = io.BytesIO()
    pil_img = Image.fromarray(np_img.astype(np.uint8))
    pil_img.save(img_byte, format='png')
    return img_byte.getvalue()


def convert_noise(np_img, noise_level):
    if noise_level == 0:
        noise_level = [0, 0]
    elif noise_level == 1:
        noise_level = [5, 25]
    elif noise_level == 2:
        noise_level = [25, 50]
    else:
        raise KeyError("Noise level should be either 0, 1, 2")

    quality = 100 - np.random.randint(*noise_level)
    img_byte = io.BytesIO()
    pil_img = Image.fromarray(np_img.astype(np.uint8))
    pil_img.save(img_byte, format='JPEG', quality=quality)
    return img_byte.getvalue()


def extract_patches(img_path, patch_size, stride, noise):
    img = imread(img_path)
    h, w, c = img.shape
    patches = []
    for i in range(0, h-patch_size+1, stride):
        for j in range(0, w-patch_size+1, stride):
            patch = img[i:i+patch_size, j:j+patch_size]

            grad = gradients(patch.astype(np.float64)/255.)
            var = np.var(patch.astype(np.float64)/255.)
            if grad >= 0.005 and var >= 0.03:
                lr_patch = skimage.transform.resize(patch, (48,48), order=3, preserve_range=True)
                hr_byte = convert(patch)
                if noise:
                    lr_byte = convert_noise(lr_patch, noise)
                else:
                    lr_byte = convert(lr_patch)
                patches.append((lr_byte, hr_byte))

    return patches


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_flist', type=str)
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--output', type=str, default='dataset/output.db')
    parser.add_argument('--table_name', type=str, default='images')
    parser.add_argument('--lr_col', type=str, default='lr_img')
    parser.add_argument('--hr_col', type=str, default='hr_img')
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--patch_size', type=int, default=48)
    parser.add_argument('--stride', type=int, default=60)
    parser.add_argument('--noise', type=int, choices=[0,1,2], default=0)
    parser.add_argument('--vacuum', action='store_true')
    args = parser.parse_args()

    conn = sqlite3.connect(args.output)
    cursor = conn.cursor()

    with conn:
        cursor.execute('PRAGMA SYNCHRONOUS = OFF')

    with conn:
        conn.execute(f'CREATE TABLE IF NOT EXISTS {args.table_name} ({args.lr_col} BLOB, {args.hr_col} BLOB)')

    if args.input_flist:
        files = [x.rstrip('\n') for x in open(args.input_flist)]
    else:
        files = glob.glob(os.path.join(args.input_dir, '*.png'))

    k = args.patch_size * args.scale
    for f in tqdm(files):
        patches = extract_patches(f, k, args.stride, args.noise)

        cursor.executemany(f'INSERT INTO {args.table_name}({args.lr_col}, {args.hr_col}) VALUES (?,?)', patches)
        conn.commit()

    cursor.execute(f'SELECT MAX(ROWID) FROM {args.table_name}')
    print(cursor.fetchone()[0])

    if args.vacuum:
        cursor.execute('VACUUM')
        conn.commit()
