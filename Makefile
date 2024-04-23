MODEL := $(if $(m),$(subst /, ,${m}),$(''))
DATASET := $(if $(d),$(subst /, ,${d}),$(''))
MODEL_PATH := $(if $(MODEL),$(word $(words ${MODEL}),1st ${MODEL})/$(lastword $(subst /, ,$(MODEL))),$(''))
DATASET_PATH := $(if $(DATASET),$(word $(words ${DATASET}),1st ${DATASET})/$(lastword $(subst /, ,$(DATASET))),$(''))

build_docker:
	sudo docker build . --tag botist

build_docker_model:
	make build_docker
	sudo docker run --runtime=nvidia --gpus all --name botist_model \
	-v ./:/root/botist \
	-v $(m)/:/root/botist/models/$(MODEL_PATH) \
	-v $(d)/:/root/botist/datasets/$(DATASET_PATH) \
	-it botist sh -c "export PYTHONPATH="${PYTHONPATH}:/root" && /root/botist/src/run_model.py --init_only"
	sudo docker commit botist_model botist:latest

run_model:
	sudo docker run --runtime=nvidia --gpus all \
	--gpus all -v ./:/root/botist \
	-v $(m)/:/root/botist/models/$(MODEL_PATH) \
	-v $(d)/:/root/botist/datasets/$(DATASET_PATH) \
	-it botist:latest sh -c "export PYTHONPATH="${PYTHONPATH}:/root" && /root/botist/src/run_model.py"

bash:
	sudo docker run --runtime=nvidia --gpus all -v ./:/root/botist -it botist bash

prune:
	sudo docker system prune -a --volumes -f