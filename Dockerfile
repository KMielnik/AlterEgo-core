FROM nvcr.io/nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04

RUN DEBIAN_FRONTEND=noninteractive apt-get -qq update \
 && DEBIAN_FRONTEND=noninteractive apt-get -qqy install python3-pip ffmpeg git libsm6 libxext6 libxrender-dev \
 && rm -rf /var/lib/apt/lists/* \
 && git clone https://github.com/KMielnik/AlterEgo-core.git \
 && pip3 install -r /AlterEgo-core/requirements.txt \
 && gdown --id 1X1iCdyghN09XaLPYCFkDMbrKnCb-hzJS -O /AlterEgo-core/model/vox-cpk.pth.tar

WORKDIR /AlterEgo-core
