MODEL := $(if $(m),$(subst /, ,${m}),$(''))
DATASET := $(if $(d),$(subst /, ,${d}),$(''))
MODEL_PATH := $(if $(MODEL),$(word $(words ${MODEL}),1st ${MODEL})/$(lastword $(subst /, ,$(MODEL))),$(''))
DATASET_PATH := $(if $(DATASET),$(word $(words ${DATASET}),1st ${DATASET})/$(lastword $(subst /, ,$(DATASET))),$(''))

build_docker:
	docker build . --tag botist

build_docker_model:
	make build_docker
	docker run --name botist_model -v ./:/root/botist \
	-v $(m)/:/root/botist/models/$(MODEL_PATH) \
	-v $(d)/:/root/botist/datasets/$(DATASET_PATH) \
	-it botist sh -c "export PYTHONPATH="${PYTHONPATH}:/root" && /root/botist/src/run_model.py --init_only"
	docker commit botist_model botist:latest

run_model:
	docker run \
	-v ./:/root/botist \
	-v $(m)/:/root/botist/models/$(MODEL_PATH) \
	-v $(d)/:/root/botist/datasets/$(DATASET_PATH) \
	-it botist:latest sh -c "export PYTHONPATH="${PYTHONPATH}:/root" && /root/botist/src/run_model.py" 

bash:
	docker run -v ./:/root/botist -it botist:latest bash

prune:
	docker system prune -a --volumes -f