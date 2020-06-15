import argparse
import os
import time

from PIL import Image
import torch

from model import SAN
from utils import ImageSplitter


def process(model, splitter, batch_size, input_fname, output_fname):
    img = Image.open(input_fname).convert('RGB')
    patches = splitter.split(img)

    if batch_size > 1:
        b = batch_size
        l = len(patches)
        patches = [torch.cat(patches[i:min(i+b, l)]) for i in range(0, l, b)]

    start_time = time.time()
    with torch.no_grad():
        out = [model(p.to(device)) for p in patches]
    print("Done in %.3f seconds!" % (time.time() - start_time))

    if batch_size > 1:
        out = [torch.split(pb, 1, dim=0) for pb in out]
        out = [p for tup in out for p in tup]

    out_img = splitter.merge(out)
    out_img.save(output_fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=int, default=2, help="scale")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size")
    parser.add_argument('--checkpoint', type=str,
                        help='The filename of pickle checkpoint.')
    parser.add_argument('--image', type=str,
                        help='The filename of image to be completed.')
    parser.add_argument('--output', default='output.png', type=str,
                        help='Where to write output.')
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    model = SAN(scale=args.scale).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    print('Model loaded')

    splitter = ImageSplitter(scale=args.scale)

    if args.image and args.output:
        process(model, splitter, args.batch_size, args.image, args.output)
    elif args.input_dir and args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        for root, dirs, files in os.walk(args.input_dir):
            for name in files:
                if any(name.endswith(ext) for ext in [".png", ".jpg", ".jpeg"]):
                    input_fname = os.path.join(root, name)
                    output_fname = os.path.join(args.output_dir, f'{os.path.splitext(name)[0]}_x{args.scale}.png')

                    process(model, splitter, args.batch_size, input_fname, output_fname)
