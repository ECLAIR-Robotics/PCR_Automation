# NOTE: please run this as 'source path/to/file/postCreateCommand.sh'
# this will allow the activation of the venv to persist after the running 
# of this script

source ./venv/bin/activate

pip install --upgrade pip
pip cache purge
pip install setuptools==65.5.0

pip install -r requirements.txt