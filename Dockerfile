FROM nvcr.io/nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04

RUN DEBIAN_FRONTEND=noninteractive apt-get -qq update \
 && DEBIAN_FRONTEND=noninteractive apt-get -qqy install python3-pip ffmpeg git libsm6 libxext6 libxrender-dev \
 && rm -rf /var/lib/apt/lists/* \
 && git clone https://github.com/KMielnik/AlterEgo-core.git

RUN pip3 install --no-cache-dir https://download.pytorch.org/whl/cu100/torch-1.0.0-cp36-cp36m-linux_x86_64.whl && \
    pip3 install --no-cache-dir -r /AlterEgo-core/requirements.txt 

RUN gdown --id 1X1iCdyghN09XaLPYCFkDMbrKnCb-hzJS -O /AlterEgo-core/model/vox-cpk.pth.tar

RUN python3 -c 'import face_alignment' && python3 -c 'face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=cpu)'

WORKDIR /AlterEgo-core
