# ml-regression-algorithm

## Setup virtual environment
To set up a virtual environment, use and deactivate, run the following commands:
```bash
python3 -m venv ./venv
source ./venv/bin/activate
deactivate
```

## Automatically Generate requirements.txt
To work in a virtual environment and have already installed the necessary libraries, use `pip freeze` to generate the `requirements.txt` file:
```bash
pip freeze > requirements.txt
```

## Install All Dependencies
To install all the dependencies listed in requirements.txt, run the following command:
```bash
pip install -r requirements.txt
```