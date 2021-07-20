echo Creating virtual environment...;
python3 -m venv ./venv;
source ./venv/bin/activate;

echo Getting all prerequiremants...;

#
pip install opencv-python;

#
pip install tensorflow;

#
pip install numpy;

echo Done!;