import subprocess
from time import time
subprocess.run("nohup /usr/bin/python3.6 /home/daniel_nlp/Lung-Cancer-Risk-Prediction/main.py > .out_" + str(time()) + ".log 2>&1 &", shell=True)