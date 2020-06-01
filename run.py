import sys
import os
from crop_video import crop_image
from crop_video import crop_video
from scipy.spatial import ConvexHull
from animate import normalize_kp
from modules.keypoint_detector import KPDetector
from modules.generator import OcclusionAwareGenerator
import moviepy.editor as mpy
from sync_batchnorm import DataParallelWithCallback
import torch
from skimage import img_as_ubyte
from skimage import img_as_float
from skimage.transform import resize
import numpy as np
import imageio
from tqdm import tqdm
from argparse import ArgumentParser
import yaml
import matplotlib
import os
import pickle
matplotlib.use('Agg')


if sys.version_info[0] < 3:
    raise Exception(
        "You must use Python 3 or higher. Recommended version is Python 3.7")


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


def make_animation(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=False):
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

        for frame_idx in tqdm(range(driving.shape[2])):
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
    for i, image in tqdm(enumerate(driving)):
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


def process_frames(source_image, driving_frames, generator, kp_detector, from_best_frame=False, adapt_movement_scale=True, gpu=False, preprocessed_kp_driving=None):
    if from_best_frame:
        i = find_best_frame(source_image, driving_frames, cpu=not gpu,
                            preprocessed_kp_driving=preprocessed_kp_driving)
        driving_forward = driving_frames[i:]
        driving_backward = driving_frames[:(i+1)][::-1]
        predictions_forward = make_animation(source_image, driving_forward, generator,
                                             kp_detector, relative=True, adapt_movement_scale=adapt_movement_scale, cpu=not gpu)
        predictions_backward = make_animation(source_image, driving_backward, generator,
                                              kp_detector, relative=True, adapt_movement_scale=adapt_movement_scale, cpu=not gpu)
        predictions = predictions_backward[::-1] + predictions_forward[1:]
    else:
        predictions = make_animation(source_image, driving_frames, generator, kp_detector,
                                     relative=True, adapt_movement_scale=adapt_movement_scale, cpu=not gpu)

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


def open_image(source_image, gpu=False, advanced_crop=False):
    source_image = imageio.imread(source_image)

    if advanced_crop:
        source_image = crop_image(source_image, gpu=opt.gpu)
    else:
        source_image = resize(source_image, (256, 256))[..., :3]

    return source_image


def preprocess_kp_driving(driving, gpu=False):
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
    for i, image in tqdm(enumerate(driving)):
        kp_driving = fa.get_landmarks(255 * image)[0]
        kp_driving = normalize_kp(kp_driving)

        preprocessed_kp_driving.append(kp_driving)

    return preprocessed_kp_driving


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

    parser.add_argument("--audio", dest="audio", action="store_true",
                        help="save original audio in result.")

    parser.add_argument("--clean_build", dest="clean", action="store_true",
                        help="do not use old temp data for video.")

    parser.set_defaults(relative=False)
    opt = parser.parse_args()

    if len(opt.source_image) != len(opt.result_video):
        print("Number of --result_video names must be equal to --source_image")
        sys.exit(1)

    if not os.path.exists("temp"):
        os.makedirs("temp")

    if not os.path.exists("output"):
        os.makedirs("output")

    video_name = os.path.splitext(opt.driving_video)[0]

    should_preprocess_video = opt.clean or not os.path.exists("temp/" + video_name) or not os.path.exists(
        "temp/" + video_name + "/driving_processed.mp4") or (opt.find_best_frame and not os.path.exists("temp/" + video_name + "/kp_driving.npy"))

    preprocessed_kp_driving = None

    if should_preprocess_video:
        import shutil
        if os.path.exists("temp/" + video_name):
            shutil.rmtree("temp/" + video_name)

        os.makedirs("temp/" + video_name)
        files_folder = "temp/" + video_name + "/"

        print("Opening video.")
        driving_video, fps = open_video(
            "videos/" + opt.driving_video, gpu=opt.gpu, advanced_crop=opt.crop)

        with imageio.get_writer(files_folder + "driving_processed.mp4", fps=fps) as writer:
            [writer.append_data(img_as_ubyte(frame))
             for frame in driving_video]

        if opt.find_best_frame:
            print("Preprocessing frames for --find_best_frame.")
            preprocessed_kp_driving = preprocess_kp_driving(
                driving_video, gpu=opt.gpu)

            with open(files_folder + "kp_driving.npy", "wb") as fp:
                pickle.dump(preprocessed_kp_driving, fp)
    else:
        files_folder = "temp/" + video_name + "/"

        print("Opening video from temp.")
        with imageio.get_reader(files_folder + "driving_processed.mp4") as reader:
            driving_video = [img_as_float(frame) for frame in reader]

        if opt.find_best_frame:
            print("Loading preprocessed frames for --find_best_frame from temp.")
            with open(files_folder + "kp_driving.npy", "rb") as fp:
                preprocessed_kp_driving = pickle.load(fp)

    print("Opening model.")
    generator, kp_detector = load_checkpoints(
        config_path=opt.config, checkpoint_path=opt.checkpoint, cpu=not opt.gpu)

    for i in range(len(opt.source_image)):
        print("Generating video for image: " +
              opt.source_image[i] + " as: " + opt.result_video[i])

        source_image = open_image(
            "images/" + opt.source_image[i], gpu=opt.gpu, advanced_crop=opt.crop)

        print("Generating video.")
        predictions = process_frames(source_image, driving_video, generator, kp_detector,
                                     from_best_frame=opt.find_best_frame, adapt_movement_scale=False, gpu=opt.gpu, preprocessed_kp_driving=preprocessed_kp_driving)

        print("Saving video.")
        with mpy.VideoFileClip("videos/" + opt.driving_video) as clip:
            frames = iter(predictions)
            with mpy.VideoClip(lambda t: img_as_ubyte(
                    next(frames)), duration=len(predictions)/clip.fps) as output:

                if opt.audio:
                    print("Copying audio.")
                    output.audio = clip.audio

                output.write_videofile(
                    "output/" + opt.result_video[i], fps=clip.fps)
