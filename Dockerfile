## ----------------------------------------------------------
## STEP 1: Build the Dockerfile: 
## docker build -t hrnet .
## ----------------------------------------------------------

FROM nvcr.io/nvidia/cuda:11.0.3-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y && apt-get install wget -y && \
        apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6 -y && \
        wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh && \
        bash Anaconda3-2021.05-Linux-x86_64.sh -b -p && \
        rm Anaconda3-2021.05-Linux-x86_64.sh && \
        ~/anaconda3/bin/conda init && \
        . ~/.bashrc && \
        ~/anaconda3/bin/conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch && \
        ~/anaconda3/bin/pip install tensorflow-gpu==2.4 && \
        ~/anaconda3/bin/pip install opencv-python==4.5.3.56 && \
        ~/anaconda3/bin/pip install ijson==3.1.4 && \
        ~/anaconda3/bin/conda install -c conda-forge yacs==0.1.6 && \
        ~/anaconda3/bin/conda install -c conda-forge json_tricks==3.15.5 -y && \
        ~/anaconda3/bin/conda install -c conda-forge tensorboardx

## ----------------------------------------------------------
## STEP 2: Start the container: 
## docker run -it --ipc=host --gpus all -v /home/gsingh/HRNet/:/root/hrnet IMAGE_ID
## ----------------------------------------------------------

## ----------------------------------------------------------
## STEP 3: Install the packages below 
## ----------------------------------------------------------

# cd ~/hrnet/code/cocoapi-master/PythonAPI && \
# /usr/bin/make && \
# ~/anaconda3/bin/python setup.py install && \
# cd ~/hrnet/code/deep-high-resolution-net.pytorch-master/lib && \
# /usr/bin/make

## ----------------------------------------------------------
## STEP 4: Docker commit container to image 
## sudo docker commit CONTAINER_ID IMAGE_NAME
## ----------------------------------------------------------

##  kill -9 $(pgrep python)
## docker ps -a
## docker stop
## docker kill