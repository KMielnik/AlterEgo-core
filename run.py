import os
import pickle
import sys
from argparse import ArgumentParser

import imageio
import numpy as np
from scipy.spatial import ConvexHull
from tqdm import tqdm

from modules.util import Range
from tool.processing import main


if sys.version_info[0] < 3:
    raise Exception(
        "You must use Python 3 or higher. Recommended version is Python 3.7")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config", default='config/vox-256.yaml', help="path to config")
    parser.add_argument("--checkpoint", default='model/vox-cpk.pth.tar',
                        help="path to checkpoint to restore")

    parser.add_argument(
        "--source_image", help="path to source image", nargs="+", required=True)
    parser.add_argument(
        "--driving_video", help="path to driving video", required=True)
    parser.add_argument(
        "--result_video", help="path to output", nargs="+", required=True)

    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true",
                        help="adapt movement scale based on convex hull of keypoints")

    parser.add_argument("--find_best_frame", dest="find_best_frame", action="store_true",
                        help="Generate from the frame that is the most alligned with source.")

    parser.add_argument("--gpu", dest="gpu",
                        action="store_true", help="add CUDA support.")

    parser.add_argument("--crop", dest="crop", action="store_true",
                        help="crop face in image and video.")

    parser.add_argument("--image_padding", dest="image_padding",type=float, choices=[Range(0.0, 1.0)], default=0.2,
                        help="how much smaller face should be in the result video (range 0.0-1.0) (only if using ---crop)")

    parser.add_argument("--audio", dest="audio", action="store_true",
                        help="save original audio in result.")

    parser.add_argument("--clean_build", dest="clean", action="store_true",
                        help="do not use old temp data for video.")

    parser.set_defaults(relative=False)
    opt = parser.parse_args()

    main(opt)
