# Installation

Installing jTWA is simple, however we only tested it on Python 3.12.1.

### Using pip
The easiest option is to install the latest package release directly from [PyPI](https://pypi.org/project/jTWA/). To that end, you may create a virtual environment
```
python -m venv env
```
and activate it
```
source env/bin/activate
```
before upgrading pip
```
pip install --upgrade pip
```
Then you can simply run
```
pip install jTWA
```
which will install jTWA along with its dependencies.

### Cloning the repository
Alternatively, you can clone the repository (maybe you also forked it first) and install it manually.
To that end you would first clone the repository using 
```
git clone https://github.com/RehMoritz/jTWA.git
```
Switch into the repository folder..
```
cd jTWA
```
..and create a virtual environment using 
```
python -m venv env
```

Activate the virtual environment with 
```
source env/bin/activate
```

Upgrade pip:
```
pip install --upgrade pip
```

Next, install the dependencies by running 
```
pip install -r requirements.txt
```