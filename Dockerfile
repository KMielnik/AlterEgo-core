FROM nvcr.io/nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

RUN DEBIAN_FRONTEND=noninteractive apt-get -qq update \
    && DEBIAN_FRONTEND=noninteractive apt-get -qqy install python3-pip ffmpeg git libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /AlterEgo-core/model
WORKDIR /AlterEgo-core

COPY requirements.txt .

RUN pip3 install --upgrade pip \
    && pip3 install --no-cache-dir https://download.pytorch.org/whl/cu100/torch-1.0.0-cp36-cp36m-linux_x86_64.whl \
    && pip3 install --no-cache-dir -r requirements.txt
RUN gdown --id 1X1iCdyghN09XaLPYCFkDMbrKnCb-hzJS -O model/vox-cpk.pth.tar
RUN python3 -c 'import face_alignment;face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device="cpu")'

COPY . /AlterEgo-core/
WORKDIR /AlterEgo-core
