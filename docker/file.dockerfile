FROM nvidia/cuda:10.2-cudnn7-runtime
# FROM continuumio/miniconda3

WORKDIR /
RUN conda install python=3.6.6 
RUN conda install pytorch=1.0 torchvision=0.2.1 -c pytorch opencv=3.3.1 -y
RUN pip install tensorboardX==1.4 scikit-image pillow=6.1.0 ipython
