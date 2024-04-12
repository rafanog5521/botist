build_docker:
	docker build . --tag botist

build_docker_tinyllama:
	docker build . --tag botist --build-arg model="Tinyllama" --progress=plain

run_model:
	docker run -v ./:/root -it botist sh -c "export PYTHONPATH="${PYTHONPATH}:/root" && ./src/run_model.py"

bash:
	docker run -v ./:/root -it botist bash

prune:
	docker system prune -a --volumes -f