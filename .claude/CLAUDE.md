## Development guide 

### Always using a venv
If you are going to run and edit code, you will always need to do it under a virtual environment.
If not already present under `.venv` (the current working folder), you can create it with
```
    uv venv --python=3.XX
```
before choosing the python version, check for any pre-existing `requirements.txt` or `pyproject.toml` or `environment.yaml` or similar files that may contain a python version, in this case we are most likely running existing code to reproduce experiments, so we need to respect the python version in order to install dependencies correctly. 
Otherwise, you can pick a safe python version that is mostly compatible with relevant packages, e.g. python 3.11.

### Handling data on the mila cluster
You need to seed the following cache paths for each console session:
```
    export HF_HOME=$SCRATCH/.cache
    export UV_CACHE_DIR=$SCRATCH/.cache
```
otherwise you will get disk quota errors, since by default caches are stored in `$HOME` and the disk is different, with a much smaller quota. If you get this problem, try to clean the home cache with `uv cache prune`.

