import face_alignment
import skimage.io
import numpy
from argparse import ArgumentParser
from skimage import img_as_ubyte
from skimage.transform import resize
from tqdm import tqdm
import os
import imageio
import numpy as np
import warnings

warnings.filterwarnings("ignore")

def extract_bbox(frame, fa):
    if max(frame.shape[0], frame.shape[1]) > 640:
        scale_factor =  max(frame.shape[0], frame.shape[1]) / 640.0
        frame = resize(frame, (int(frame.shape[0] / scale_factor), int(frame.shape[1] / scale_factor)))
        frame = img_as_ubyte(frame)
    else:
        scale_factor = 1
    frame = frame[..., :3]
    bboxes = fa.face_detector.detect_from_image(frame[..., ::-1])
    if len(bboxes) == 0:
        return []
    return np.array(bboxes)[:, :-1] * scale_factor



def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def join(tube_bbox, bbox):
    xA = min(tube_bbox[0], bbox[0])
    yA = min(tube_bbox[1], bbox[1])
    xB = max(tube_bbox[2], bbox[2])
    yB = max(tube_bbox[3], bbox[3])
    return (xA, yA, xB, yB)


def compute_bbox(start, end, tube_bbox, frame_shape, image_shape, increase_area):
    left, top, right, bot = tube_bbox
    width = right - left
    height = bot - top

    #Computing aspect preserving bbox
    width_increase = max(increase_area, ((1 + 2 * increase_area) * height - width) / (2 * width))
    height_increase = max(increase_area, ((1 + 2 * increase_area) * width - height) / (2 * height))

    left = int(left - width_increase * width)
    top = int(top - height_increase * height)
    right = int(right + width_increase * width)
    bot = int(bot + height_increase * height)

    top, bot, left, right = max(0, top), min(bot, frame_shape[0]), max(0, left), min(right, frame_shape[1])

    return {
        "start": start,
        "end": end,
        "bbox": (left, right, top, bot),
        "scale": image_shape
    }


def compute_bbox_trajectories(trajectories, frame_shape, increase, min_frames, image_shape):
    commands = []
    for i, (bbox, tube_bbox, start, end) in enumerate(trajectories):
        if (end - start) > min_frames:
            command = compute_bbox(start, end, tube_bbox, frame_shape, image_shape=image_shape, increase_area=increase)
            commands.append(command)
    return commands


def process_video_for_crop(video, gpu=False, increase=0.1, min_frames = 5, image_shape=(256,256)):
    device = 'cuda' if gpu else 'cpu'

    iou_with_initial = 0.25

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device)

    trajectories = []
    previous_frame = None
    commands = []
    try:
        for i, frame in tqdm(enumerate(video)):
            frame_shape = frame.shape
            bboxes =  extract_bbox(frame, fa)
            ## For each trajectory check the criterion
            not_valid_trajectories = []
            valid_trajectories = []

            for trajectory in trajectories:
                tube_bbox = trajectory[0]
                intersection = 0
                for bbox in bboxes:
                    intersection = max(intersection, bb_intersection_over_union(tube_bbox, bbox))
                if intersection > iou_with_initial:
                    valid_trajectories.append(trajectory)
                else:
                    not_valid_trajectories.append(trajectory)

            commands += compute_bbox_trajectories(not_valid_trajectories, frame_shape, increase, min_frames, image_shape)
            trajectories = valid_trajectories

            ## Assign bbox to trajectories, create new trajectories
            for bbox in bboxes:
                intersection = 0
                current_trajectory = None
                for trajectory in trajectories:
                    tube_bbox = trajectory[0]
                    current_intersection = bb_intersection_over_union(tube_bbox, bbox)
                    if intersection < current_intersection and current_intersection > iou_with_initial:
                        intersection = bb_intersection_over_union(tube_bbox, bbox)
                        current_trajectory = trajectory

                ## Create new trajectory
                if current_trajectory is None:
                    trajectories.append([bbox, bbox, i, i])
                else:
                    current_trajectory[3] = i
                    current_trajectory[1] = join(current_trajectory[1], bbox)


    except IndexError as e:
        raise (e)

    commands += compute_bbox_trajectories(trajectories, frame_shape, increase, min_frames, image_shape)
    return commands

def crop_video(video, gpu=False, increase=0.1, min_frames = 5, image_shape=(256,256)):
    if len(video) == 0:
        return video

    commands = process_video_for_crop(video, gpu=gpu, increase=increase, min_frames=min_frames, image_shape=image_shape)

    output = []

    for command in commands:
        for i in tqdm(range(command["start"], command["end"]+1)):
            bbox = command["bbox"]
            frame = video[i][bbox[2]:bbox[3],bbox[0]:bbox[1],:]

            frame = resize(frame, image_shape)[..., :3]

            output.append(frame)

    return output

def crop_image(image, gpu=False, increase=0.1, min_frames = -1, image_shape=(256,256)):
    return crop_video([image], gpu=gpu, increase=increase, min_frames = min_frames, image_shape=image_shape)[0]

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--image_shape", default=(256, 256), type=lambda x: tuple(map(int, x.split(','))),
                        help="Image shape")
    parser.add_argument("--increase", default=0.1, type=float, help='Increase bbox by this amount')
    parser.add_argument("--iou_with_initial", type=float, default=0.25, help="The minimal allowed iou with inital bbox")
    parser.add_argument("--inp", required=True, help='Input image or video')
    parser.add_argument("--min_frames", type=int, default=1,  help='Minimum number of frames')
    parser.add_argument("--gpu", dest="gpu", action="store_true", help="gpu mode.")

    args = parser.parse_args()

    video = imageio.get_reader(args.inp)

    video = [frame for frame in video]

    commands = process_video_for_crop(video, min_frames=args.min_frames, gpu=args.gpu, increase=args.increase, image_shape=args.image_shape)

    for command in commands:
        print (command)
