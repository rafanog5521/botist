FROM ubuntu:22.04
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

#Update packages
RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get install -y curl python3 python3.10-venv pip git

WORKDIR /root
#Clone repository
RUN git clone https://github.com/rafanog5521/botist.git && cd botist && git checkout master -f && git pull

#Install python dependencies
WORKDIR /root/botist
RUN python3 -m venv env && source ./env/bin/activate && pip install -r ./requirements.txt 

#Run model
RUN source ./env/bin/activate && export PYTHONPATH="${PYTHONPATH}:/root/botist/" && ./src/run_model.py