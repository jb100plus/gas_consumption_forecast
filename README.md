# gas_consumption_forecast
gas consumption forecast

this is how to get the model running with ubuntu18.04:
```
mkdir test
cd test
git clone https://github.com/jb100plus/gas_consumption_forecast.git
cd gas_consumption_forecast
mkdir data
pip3 install --upgrade pip
pip3 install virtualenv
virtualenv venv
source venv/bin/activate
pip install pandas
pip install keras
pip install tensorflow==1.10
pip install sklearn
pip install matplotlib
pip install seaborn
sudo apt install python-tk

python trainmodel.py
```
