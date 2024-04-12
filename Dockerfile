FROM ubuntu:22.04
RUN rm /bin/sh && ln -s /bin/bash /bin/sh
ARG model


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

#Setup model (if needed)
RUN 

if [[ -z "$model" ]]; then 
    echo Model not provided;  
else if [[ $model == "Tinyllama" ]]; then 
    echo Argument not provided; 
    fi;
fi