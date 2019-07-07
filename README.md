# pid-tuning-ai

PID Control tuning with Artificial Intelligence in Python

Method :
- Genetic Algorithm for creating dataset
- Neural Network for value approximator

Depedencies :
- Python 2
- Numpy

Files :
- main.py for main script
- ai_ga.py for Genetic Algorithm
- ai_nn.py for Neural Network
- step_info.py for getting Step info (Overshoot, Risetime etc.) for creating objective function
- file_model.py for saving/load dataset from Genetic Algorithm

Usage :
- set the protocol connection between pc and microcontroller like arduino script in microcontroller folder
- edit objective and value data from ai_ga.py script
- make sure all electronic devices ready (wiring and calibration)
- run microcontroller first
- run main.py
