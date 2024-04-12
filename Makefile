build_docker:
	docker build . --tag botist

build_docker_tinyllama:
	docker build . --tag botist --build-arg model="Tinyllama" --progress=plain

run_model:
	docker run -v ./:/root/botist -it botist sh -c "source ./py_env/bin/activate && export PYTHONPATH="${PYTHONPATH}:/root/botist" && ./botist/src/run_model.py"