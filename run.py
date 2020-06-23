import os
import pickle
import concurrent.futures
import sys
from argparse import ArgumentParser
import asyncio
from modules.util import Range

from tool.processing import process_task
from task import Task
import json


if sys.version_info[0] < 3:
    raise Exception(
        "You must use Python 3 or higher. Recommended version is Python 3.7")

        
async def main(task: Task):
    if not os.path.exists("temp"):
        os.makedirs("temp")

    if not os.path.exists("output"):
        os.makedirs("output")

    async for event in process_task(task):
        if event.EventType.isError:
            print("ERROR: " + event.EventType.text + " Task time: " + "{:.2f}s".format(event.Time) + " CANCELING TASK!")
        else:
            print(event.EventType.text + " Time: " + "{:.2f}s".format(event.Time))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config", default='config/vox-256.yaml', help="path to config")
    parser.add_argument("--checkpoint", default='model/vox-cpk.pth.tar',
                        help="path to checkpoint to restore")

    parser.add_argument(
        "--source_images", help="paths to source images", nargs="+", required=True)
    parser.add_argument(
        "--driving_video", help="path to driving video", required=True)
    parser.add_argument(
        "--result_videos", help="path to output", nargs="+", required=True)

    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true",
                        help="adapt movement scale based on convex hull of keypoints")

    parser.add_argument("--find_best_frame", dest="find_best_frame", action="store_true",
                        help="Generate from the frame that is the most alligned with source.")

    parser.add_argument("--gpu", dest="gpu",
                        action="store_true", help="add CUDA support.")

    parser.add_argument("--crop", dest="crop", action="store_true",
                        help="crop face in image and video.")

    parser.add_argument("--image_padding", dest="image_padding", type=float, choices=[Range(0.0, 1.0)], default=0.2,
                        help="how much smaller face should be in the result video (range 0.0-1.0) (only if using ---crop)")

    parser.add_argument("--audio", dest="audio", action="store_true",
                        help="save original audio in result.")

    parser.add_argument("--clean_build", dest="clean_build", action="store_true",
                        help="do not use old temp data for video.")

    parser.set_defaults(relative=False)
    opt = parser.parse_args()

    task = Task()

    task.adapt_scale = opt.adapt_scale
    task.audio = opt.audio
    task.checkpoint = opt.checkpoint
    task.clean_build = opt.clean_build
    task.config = opt.config
    task.crop = opt.crop
    task.driving_video = opt.driving_video
    task.find_best_frame = opt.driving_video
    task.gpu = opt.gpu
    task.image_padding = opt.image_padding
    task.result_videos = opt.result_videos
    task.source_images = opt.source_images

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main(task))
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
