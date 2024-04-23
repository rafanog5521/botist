FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

#Update packages
RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get install -y curl python3 pip git

WORKDIR /root
#Clone repository
#RUN git clone https://github.com/rafanog5521/botist.git
COPY ./ ./botist

#Install python dependencies
WORKDIR /root/botist
RUN pip install -r ./requirements.txt