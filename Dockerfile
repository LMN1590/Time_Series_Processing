FROM amazon/aws-lambda-python:3.10

WORKDIR /app

ARG DEBIAN_FRONTEND=noninteractive

RUN yum update -y \
&& yum update -y python3 curl libcom_err ncurses expat libblkid libuuid libmount \
&& yum install python3-pip git ffmpeg libsm6 libxext6 -y \
&& yum clean all \
&& rm -rf /var/cache/yum

COPY ./requirements_docker.txt ./requirements_docker.txt

RUN pip3 install --default-timeout=100 -r requirements.txt --no-cache-dir \
&& rm -rf /root/.cache