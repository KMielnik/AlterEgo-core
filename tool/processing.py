import asyncio
import concurrent.futures
import os
import pickle
import sys
from typing import List

import imageio
import moviepy.editor as mpy
import numpy as np
import torch
import yaml
from scipy.spatial import ConvexHull
from skimage import img_as_float, img_as_ubyte
from skimage.transform import resize
from tqdm import tqdm

from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from sync_batchnorm import DataParallelWithCallback
from task import Task
from tool.animate import normalize_kp
from tool.crop_video import crop_image, crop_video


def load_checkpoints(config_path, checkpoint_path, cpu=False):

    with open(config_path) as f:
        config = yaml.load(f)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    if not cpu:
        generator.cuda()

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not cpu:
        kp_detector.cuda()

    if cpu:
        checkpoint = torch.load(
            checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)

    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])

    if not cpu:
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()

    return generator, kp_detector


def make_animation(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=False, progress_bar=None):
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(
            np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(
            np.float32)).permute(0, 4, 1, 2, 3)
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in range(driving.shape[2]):
            progress_bar.update(1)
            driving_frame = driving[:, :, frame_idx]
            if not cpu:
                driving_frame = driving_frame.cuda()
            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

            predictions.append(np.transpose(
                out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions


def find_best_frame(source, driving, cpu=False, preprocessed_kp_driving=None):
    import face_alignment

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                      device='cpu' if cpu else 'cuda')
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm = float('inf')
    frame_num = 0
    for i, image in enumerate(driving):
        if preprocessed_kp_driving != None:
            kp_driving = preprocessed_kp_driving[i]
        else:
            kp_driving = fa.get_landmarks(255 * image)[0]
            kp_driving = normalize_kp(kp_driving)

        new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
        if new_norm < norm:
            norm = new_norm
            frame_num = i
    return frame_num


def process_frames(source_image, driving_frames, generator, kp_detector, from_best_frame=False, adapt_movement_scale=True, gpu=False, preprocessed_kp_driving=None, h_progress=True):
    frames_to_process = len(driving_frames) + \
        1 if from_best_frame else len(driving_frames)
    progress_bar = tqdm(total=frames_to_process)

    if from_best_frame:
        i = find_best_frame(source_image, driving_frames, cpu=not gpu,
                            preprocessed_kp_driving=preprocessed_kp_driving)
        driving_forward = driving_frames[i:]
        driving_backward = driving_frames[:(i+1)][::-1]
        predictions_forward = make_animation(source_image, driving_forward, generator,
                                             kp_detector, relative=True, adapt_movement_scale=adapt_movement_scale, cpu=not gpu, progress_bar=progress_bar)
        predictions_backward = make_animation(source_image, driving_backward, generator,
                                              kp_detector, relative=True, adapt_movement_scale=adapt_movement_scale, cpu=not gpu, progress_bar=progress_bar)
        predictions = predictions_backward[::-1] + predictions_forward[1:]
    else:
        predictions = make_animation(source_image, driving_frames, generator, kp_detector,
                                     relative=True, adapt_movement_scale=adapt_movement_scale, cpu=not gpu, progress_bar=progress_bar)
    progress_bar.close()
    return predictions


def open_video(source_video, gpu=False, advanced_crop=False):
    with imageio.get_reader(source_video) as reader:
        driving_video = [frame for frame in reader]
        fps = reader.get_meta_data()['fps']

    if advanced_crop:
        driving_video = crop_video(driving_video, gpu=gpu, min_frames=1)
    else:
        driving_video = [resize(frame, (256, 256))[..., :3]
                         for frame in driving_video]

    return driving_video, fps


def open_image(source_image, gpu=False, advanced_crop=False, increase=0.2):
    source_image = imageio.imread(source_image)

    if advanced_crop:
        source_image = crop_image(source_image, gpu=gpu, increase=increase)
    else:
        source_image = resize(source_image, (256, 256))[..., :3]

    return source_image


def preprocess_kp_driving(driving: List, gpu=False, h_progress=True):
    import face_alignment

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                      device='cuda' if gpu else 'cpu')

    preprocessed_kp_driving = []
    driving_enumerator = enumerate(
        tqdm(driving)) if h_progress else enumerate(driving)
    for i, image in driving_enumerator:
        kp_driving = fa.get_landmarks(255 * image)[0]
        kp_driving = normalize_kp(kp_driving)

        preprocessed_kp_driving.append(kp_driving)

    return preprocessed_kp_driving


async def process_task(task: Task):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        loop = asyncio.get_event_loop()

        video_name = os.path.splitext(task.driving_video)[0]

        should_preprocess_video = task.clean_build or not os.path.exists("temp/" + video_name) or not os.path.exists(
            "temp/" + video_name + "/driving_processed.mp4") or (task.find_best_frame and not os.path.exists("temp/" + video_name + "/kp_driving.npy"))

        preprocessed_kp_driving = None

        if should_preprocess_video:
            import shutil
            if os.path.exists("temp/" + video_name):
                shutil.rmtree("temp/" + video_name)

            os.makedirs("temp/" + video_name)
            files_folder = "temp/" + video_name + "/"

            yield "Opening video."
            driving_video, fps = await loop.run_in_executor(
                pool, open_video,
                "videos/" + task.driving_video,
                task.gpu,
                task.crop)

            with imageio.get_writer(files_folder + "driving_processed.mp4", fps=fps) as writer:
                [writer.append_data(img_as_ubyte(frame))
                 for frame in driving_video]

            if task.find_best_frame:
                yield "Preprocessing frames for --find_best_frame."
                preprocessed_kp_driving = await loop.run_in_executor(
                    pool, preprocess_kp_driving,
                    driving_video, task.gpu)

                with open(files_folder + "kp_driving.npy", "wb") as fp:
                    pickle.dump(preprocessed_kp_driving, fp)
        else:
            files_folder = "temp/" + video_name + "/"

            yield "Opening video from temp."
            with imageio.get_reader(files_folder + "driving_processed.mp4") as reader:
                driving_video = [img_as_float(frame) for frame in reader]

            if task.find_best_frame:
                yield "Loading preprocessed frames for --find_best_frame from temp."
                with open(files_folder + "kp_driving.npy", "rb") as fp:
                    preprocessed_kp_driving = pickle.load(fp)

        yield "Opening model."
        generator, kp_detector = await loop.run_in_executor(
            pool, load_checkpoints,
            task.config,
            task.checkpoint,
            not task.gpu)

        for i in range(len(task.source_images)):
            yield ("Generating video for image: " +
                   task.source_images[i] + " as: " + task.result_videos[i])

            source_image = await loop.run_in_executor(
                pool, open_image,
                "images/" + task.source_images[i],
                task.gpu,
                task.crop,
                task.image_padding)

            yield "Generating video."
            predictions = await loop.run_in_executor(
                pool, process_frames,
                source_image,
                driving_video,
                generator,
                kp_detector,
                task.find_best_frame,
                False,
                task.gpu,
                preprocessed_kp_driving)

            yield "Saving video."
            with mpy.VideoFileClip("videos/" + task.driving_video) as clip:
                frames = iter(predictions)
                with mpy.VideoClip(lambda t: img_as_ubyte(
                        next(frames)), duration=len(predictions)/clip.fps) as output:

                    if task.audio:
                        yield "Copying audio."
                        output.audio = clip.audio

                    output.write_videofile(
                        "output/" + task.result_videos[i], fps=clip.fps, verbose=False, logger=None)


async def main(task: Task):
    if not os.path.exists("temp"):
        os.makedirs("temp")

    if not os.path.exists("output"):
        os.makedirs("output")

    async for message in process_task(task):
        print(message)
