build_docker:
	docker build . --tag botist

run_model:
	docker exec -it botist sh -c "./src/run_model.py"