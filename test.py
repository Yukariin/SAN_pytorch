import argparse

from PIL import Image
import torch

from model import SAN
from utils import ImageSplitter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=int, default=2, help="scale")
    parser.add_argument('--checkpoint', type=str,
                        help='The filename of pickle checkpoint.')
    parser.add_argument('--image', type=str,
                        help='The filename of image to be completed.')
    parser.add_argument('--output', default='output.png', type=str,
                        help='Where to write output.')
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    model = SAN(scale=args.scale).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    print('Model loaded')

    splitter = ImageSplitter(scale=args.scale)

    img = Image.open(args.image).convert('RGB')
    patches = splitter.split(img)

    import time
    start_time = time.time()
    with torch.no_grad():
        out = [model(p.to(device)) for p in patches]
    print("Done in %.3f seconds!" % (time.time() - start_time))

    out_img = splitter.merge(out)
    out_img.save(args.output)
