FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu16.04
LABEL authors="Sina Masoud-Ansari, Tabish Rashid modidied by max bleggi"

ENV SC2PATH /pymarl/3rdparty/StarCraftII

RUN apt-get update && apt-get install -y \
	git

COPY requirements.txt .

# Install python3
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    python3.8 python3-pip python3.8-dev
RUN pip3 install --upgrade pip setuptools
RUN pip3 install -r requirements.txt

WORKDIR /pymarl
