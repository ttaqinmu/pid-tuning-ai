import os
import json
import csv
import numpy as np
from matplotlib import pyplot as plt

class FileModel:
    '''
    This class for opening or saving model of population or properties from genetic AI
    '''
    def __init__(self, model_name):
        self.dict = os.getcwd()

        self.model_name = model_name
        self.name_population = self.model_name + '_population.csv'
        self.name_properties = self.model_name + '_properties.json'

        os.system("mkdir -p genetic_data/"+self.model_name)
        os.system("mkdir -p genetic_data/"+self.model_name+"/graph")
        os.system("mkdir -p genetic_data/"+self.model_name+"/step")

    def save_population_to_model(self, population):
        temp_population = []

        for i in range(len(population)):
            temp_population.append(i)
            temp_population[i] = [ 
                population[i].kp,
                population[i].ki,
                population[i].kd,
                population[i].fitness,
                population[i].risetime,
                population[i].overshoot,
                population[i].settling_time,
                population[i].peak,
                population[i].steadystate,
                population[i].creator,
                population[i].saved
            ]

        with open(self.dict + "/genetic_data/" + self.model_name + "/" + self.name_population, 'w') as population_file:
            population_writer= csv.writer(population_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for i in range(len(population)):
                population_writer.writerow(temp_population[i])

    def save_properties_to_model(self, properties):
        with open(self.dict + "/genetic_data/" + self.model_name + "/" + self.name_properties, 'w') as fp:
            json.dump(properties, fp)
    
    def save_individu_to_model(self, individu):
        temp_population = [ 
            individu.kp,
            individu.ki,
            individu.kd,
            individu.fitness,
            individu.risetime,
            individu.overshoot,
            individu.settling_time,
            individu.peak,
            individu.steadystate,
            individu.creator,
            individu.saved
        ]
        os.system("mkdir -p genetic_data/"+self.model_name)
        with open(self.dict + "/genetic_data/" + self.model_name + "/" + self.name_population, 'a') as population_file:
            population_writer= csv.writer(population_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            population_writer.writerow(temp_population)
    
    def save_individu_to_graph(self, x_list, y_list, setpoint, name):
        sp_list = np.full(len(x_list), setpoint)
        plt.figure(name)
        plt.plot(x_list, y_list, label="control data")
        plt.plot(x_list, sp_list, label="setpoint")
        plt.savefig(self.dict + "/genetic_data/" + self.model_name + "/graph/" + str(name) + ".png")
        plt.close(name)

        temp_step= []

        for i in range(len(x_list)):
            temp_step.append(i)
            temp_step[i] = [ 
                x_list[i],
                y_list[i]
            ]
        
        with open(self.dict + "/genetic_data/" + self.model_name + "/step/" + str(name) + ".csv", 'w') as step_file:
            step_writer = csv.writer(step_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for i in range(len(x_list)):
                step_writer.writerow(temp_step[i])


    def open_population_from_model(self):
        population_temp = []
        fol = self.dict + "/genetic_data/" + self.model_name + "/" + self.name_population
        with open(fol) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                population_temp.append(1)
                population_temp[line_count] = [
                    row[0],
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                    row[5],
                    row[6],
                    row[7],
                    row[8],
                    row[9],
                    row[10]
                ]
                line_count += 1
        return population_temp
    
    def open_properties_from_model(self):
        fol = self.dict + "/genetic_data/" + self.model_name + "/" + self.name_properties
        with open(fol) as json_file:
            pooldata = json.load(json_file)
        return pooldata