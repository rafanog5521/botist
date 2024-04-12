build_docker:
	docker build . --tag botist

build_docker_tinyllama
	docker build . --tag botist 

run_model:
	docker run -it botist sh -c "source ./env/bin/activate && export PYTHONPATH="${PYTHONPATH}:/root/botist/" && ./src/run_model.py"