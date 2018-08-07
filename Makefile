make venv:
	# this isn't under CI so may explode
	conda create -n pint-pandas python=3.6
	conda activate pint-pandas; \
		conda install -c conda-forge pandas; \
		pip install --no-deps -e .
