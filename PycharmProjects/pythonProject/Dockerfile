FROM nvidia/cuda:10.2-base
CMD nvidia-smi

MAINTAINER Kolesov Aleksandr "kolesov.as@phystech.edu"

FROM pytorch/pytorch
RUN apt-get update
RUN apt-get install apt-file -y
RUN apt-file update
RUN apt-get install -y git build-essential curl wget software-properties-common zip unzip vim
EXPOSE 7000
EXPOSE 5000

# copy data files

COPY requirements.txt /home/requirements.txt
WORKDIR /home/

RUN pip install --upgrade pip
RUN pip install  -r ./requirements.txt

 Cd