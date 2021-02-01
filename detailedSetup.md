
### Detailed Set-up


#### Pull from GitHub
go to local folder where you want the files and type 
```
git init
```

copy all files from the remote repository into this local folder:
```
git pull git@github.com:donken/NeuralOC
```

Set vim as the default editor (this step often just helps on linux):
```
git config --global core.editor "vim"
```

#### Virtual Environment

We used python 3.6.9, which will influence the package versions.
You can check what version of python you have using
```
python --version
python3 --version
```

Create a virtual environment (may need to install virtualenv command) to hold all the python package versions for this project:
```
virtualenv -p python3 ocEnv
```
where python3 is used after you have checked the version above.


Start up the virtual environment:
```
source ocEnv/bin/activate
```

Install all the requirements:
```
pip install -r requirements.txt 
```

Try a simple command
```
python trainOC.py
```
