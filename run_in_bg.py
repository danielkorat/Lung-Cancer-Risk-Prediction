import subprocess
subprocess.run("nohup /usr/bin/python3.6 /home/daniel_nlp/Lung-Cancer-Risk-Prediction/main.py > out.log 2>&1 &", shell=True)