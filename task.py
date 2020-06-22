from typing import List

class Task:

    source_images: List[str]
    driving_video: str
    result_video: List[str]

    config: str = 'config/vox-256.yaml'
    checkpoint: str = 'model/vox-cpk.pth.tar'

    gpu: bool

    crop: bool
    image_padding: float

    audio: bool

    find_best_frame: bool

    adapt_scale: bool

    clean_build: bool
