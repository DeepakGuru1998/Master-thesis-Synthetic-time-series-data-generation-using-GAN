environment:
	python3.6 -m venv environment

requirements: environment
	( \
		source environment/bin/activate; \
		pip install --upgrade pip; \
		python -m pip install -r requirements.txt; \
		deactivate; \
	)

# Upload environment to HDFS
upload_env: requirements
	( \
		source environment/bin/activate; \
		venv-pack; \
		hdfs dfs -put -f environment.tar.gz /data/user/$(USER)/; \
		rm environment.tar.gz; \
		deactivate; \
	)
