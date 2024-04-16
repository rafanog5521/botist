build_docker:
	docker build . --tag botist

build_docker_model:
	make build_docker
	docker run --name botist_model -v ./:/root/botist -it botist sh -c "export PYTHONPATH="${PYTHONPATH}:/root" && /root/botist/src/run_model.py --init_only"
	docker commit botist_model botist:latest

run_model:
	docker run -v ./:/root/botist -it botist:latest sh -c "export PYTHONPATH="${PYTHONPATH}:/root" && /root/botist/src/run_model.py"

bash:
	docker run -v ./:/root/botist -it botist:latest bash

prune:
	docker system prune -a --volumes -f