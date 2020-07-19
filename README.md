# AlterEgo-core

### This repository is based on:  [first-order-model](https://github.com/AliaksandrSiarohin/first-order-model)

This is a modification of first-order-model, which has been modified to focus on VoxCeleb dataset model.

Some options have been added, like an option to retain audio, automatic cropping, saving temp files that allow faster generation of successive animations based on the same driving video. 

All of it can be used with one command.


## How to use

### Native

1. Clone the repository
1. Place pretrained model vox-cpk.pth.tar in /model directory (link from my gdrive is available in /model)
1. Place images/videos in corresponding folders
1. ```python run.py``` with parameters (example below)

### Docker

1. Build the image with Dockerfile ``` docker build -t {imagename} {directory_with_Dockerfile} ```
1. ``` docker run -it --rm {images/videos/output/temp folders mounted as volumes} {imagename} {python command} ```

### Python ```run.py``` parameters

You only need to specify filename with extension, for example ``` video.mp4 ```

* ``` --driving_video {filename} ``` filename of driving video - REQUIRED
* ``` --source_image {filenames} ``` filenames of images to animate, space delimeter - REQUIRED >=1
* ``` --result_video {filenames} ``` filenames of generated videos, space delimeter - REQUIRED >=1 AND SAME QUANTITY AS ``` --source_image ```
* ``` --gpu ``` enables CUDA support
* ``` --crop ``` enables croping instead of rescale (worse performance, definitely worth it, unless you cropped square image/video yourself)
* ``` --image_padding ``` changes image padding (default=0.2), 0 means face is covering all of result video, 1 would be maximum zoom out (works only if using ``` --crop ```)
* ``` --find_best_frame ``` finds the best frame to start generating video from
* ``` --audio ``` retain original audio from input video
* ``` --clean_build ``` do not use previous cropped video data
* ``` --api ``` outputs json events, instead of human readable ones. Specification of event is available in [output_event.py](output_event.py) with all possible states.

### Full example with docker

```
docker run -it --rm --gpus all \
    -v $HOME/AlterEgo-core-data/images:/AlterEgo-core/images \
    -v $HOME/AlterEgo-core-data/videos:/AlterEgo-core/videos \
    -v $HOME/AlterEgo-core-data/output:/AlterEgo-core/output \
    -v $HOME/AlterEgo-core-data/temp:/AlterEgo-core/temp \
    kamilmielnik/alterego-core:2.0.3 \
    python3 run.py \
        --driving_video source.mp4 \
        --source_image a.jpg b.jpg c.jpg \
        --result_video resultA.mp4 resultB.mp4 resultC.mp4 \
        --gpu \
        --find_best_frame \
        --crop \
        --image_padding 0.2 \
        --audio \
        --clean_build
```

### Colab Demo 
To try this repo, you can use the demo: [```demo-colab.ipynb```](https://colab.research.google.com/github/KMielnik/AlterEgo-core/blob/master/demo-colab.ipynb).


#### Additional notes

Citation for first-order-model:

```
@InProceedings{Siarohin_2019_NeurIPS,
  author={Siarohin, Aliaksandr and Lathuilière, Stéphane and Tulyakov, Sergey and Ricci, Elisa and Sebe, Nicu},
  title={First Order Motion Model for Image Animation},
  booktitle = {Conference on Neural Information Processing Systems (NeurIPS)},
  month = {December},
  year = {2019}
}
```