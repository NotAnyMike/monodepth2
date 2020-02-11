FROM nvidia/cuda:10.2-cudnn7-runtime

# Installing conda
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

# Installing dependancies
RUN conda install python=3.6.6 -y
RUN conda install pytorch=1.0 torchvision=0.2.1 opencv=3.3.1 -c pytorch -y
RUN pip install tensorboardX==1.4 scikit-image pillow==6.1.0 ipython

WORKDIR /monodepth2
