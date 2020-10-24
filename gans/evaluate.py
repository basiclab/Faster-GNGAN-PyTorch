import argparse
import glob
import os

from PIL import Image
from torchvision.transforms.functional import to_tensor

from common.score.score import get_inception_and_fid_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Calculate FID and inception score")
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--fid_cache', type=str, required=True)
    args = parser.parse_args()

    files = (
        list(glob.glob(os.path.join(args.dir, '*.png'))) +
        list(glob.glob(os.path.join(args.dir, '*.jpg')))
    )

    def image_loader(files):
        for file_path in files:
            img = Image.open(file_path)
            yield to_tensor(img).numpy()
    loader = image_loader(files)

    (IS, IS_std), FID = get_inception_and_fid_score(
        loader, args.fid_cache, num_images=len(files),
        verbose=True, use_torch=True)
    print("IS: %6.3f(%.3f), FID: %7.3f" % (IS, IS_std, FID))
