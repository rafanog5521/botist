FROM ubuntu:22.04
RUN rm /bin/sh && ln -s /bin/bash /bin/sh
ARG model

#Update packages
RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get install -y curl python3 python3.10-venv pip git

WORKDIR /root
#Copy your repository
COPY ./ ./

#Install python dependencies
RUN python3 -m venv py_env && source ./py_env/bin/activate && pip install -r ./requirements.txt

#Setup model (if needed)
RUN \
if [[ -z "$model" ]]; then \
    echo Model not provided; \  
else if [[ $model == "Tinyllama" ]]; then \ 
    source ./py_env/bin/activate && \
    export PYTHONPATH="${PYTHONPATH}:/root/" && \
    ./src/run_model.py --init_only; \
    fi; \
fi