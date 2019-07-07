import numpy as np
import random
import time
from scipy.ndimage.filters import gaussian_filter1d
from step_info import StepInfo

class DNA:
    '''
    This is class for DNA to store properties value or do a action for population object
    '''
    def __init__(self, kp, ki, kd):
        # Store properties of DNA's individu
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.fitness = 0
        self.normalize_prob = 0

        self.risetime = 0
        self.overshoot = 0
        self.settling_time = 0
        self.peak = 0
        self.steadystate = 0

        self.saved = 0
        self.creator = "random"

        self.x_step = np.array([])
        self.y_step = np.array([])

        self.serial = None

    def calculate_fitness(self, max_step, sp, serial):
        # calculate fitness from step control of PID
        self.serial = serial

        #change this with communication protocol
        self.serial.write("k "+ str(self.kp) + " "+ str(self.ki) + " "+ str(self.kd) + " "+ str(sp) + " "+ str(max_step))

        print "\t\tKP:",self.kp," KI:",self.ki," KD:",self.kd
        print "\t\tSetpoint:",sp

        x_list = []
        y_list = []

        step = 0
        data = 0

        while len(y_list) < max_step:
            data_serial = self.serial.readline()
            data_serial = data_serial.replace('\r','')
            data_serial = data_serial.replace('\n','')
            try:
                data = float(data_serial)
            except ValueError:
                data = data

            step += 1

            x_list.append(step)
            y_list.append(data)
            print "\t\t\tStep:",step," Height:",data

        #smoothing graph plot of y value
        ysmoothed = gaussian_filter1d(y_list, sigma=2)

        #finding step info from each iteration
        info = StepInfo(x_list, ysmoothed, sp)

        self.risetime = info.getRiseTime()
        self.overshoot = info.getOvershoot()
        self.peak = info.getPeak()
        self.settling_time = info.getSettlingTime()
        self.steadystate = info.getSteadyStateError()

        self.y_step = ysmoothed
        self.x_step = x_list

        if self.settling_time == 0:
            hitung_settling = len(y_list)
        else:
            hitung_settling = self.settling_time

        #fitness function calculating from step info
        #self.fitness = 100/(self.risetime+(self.overshoot*self.overshoot)+self.peak+hitung_settling+self.steadystate)
        self.fitness = 1.0/info.getMSE()

        print "\t\tRiseTime:",self.risetime," Overshoot:",self.overshoot," Peak:",self.peak," SettlingTime:",self.settling_time," Steadystate Error:", self.steadystate
        print "\t\tFitness:",self.fitness

        time.sleep(5)


class Population:
    '''
    This class contains individual and evolution function.
    Main class for genetic algorithm
    '''
    population = []
    max = False

    def __init__(self, mutation_rate = 0.3, crossover_rate = 0.7,
                    max_population = 100, max_timestep = 10, max_gain_value = 1, min_gain = 1,max_gain = 10,
                    max_generate_initial_population = 100, setpoint = 30, serial = None):

        self.serial = serial

        self.properties = {
            "MutationRate" : mutation_rate,
            "CrossoverRate" : crossover_rate,
            "MaxPopulation" : max_population,
            "MaxTimestep" : max_timestep,
            "MinGain" : min_gain,
            "MaxGain" : max_gain,
            "MaxGenerateInitial" : max_generate_initial_population,
            "MaxGainValue" : max_gain_value,
            "SetPoint" : setpoint
        }

        if mutation_rate != None:
            self.mutation_rate = mutation_rate
            self.crossover_rate = crossover_rate
            self.max_population = max_population
            self.max_timestep = max_timestep
            self.min_gain = min_gain
            self.max_gain = max_gain
            self.max_generate_initial_population = max_generate_initial_population
            self.max_gain_value = max_gain_value
            self.setpoint = setpoint

            self.properties = {
                "MutationRate" : self.mutation_rate,
                "CrossoverRate" : self.crossover_rate,
                "MaxPopulation" : self.max_population,
                "MaxTimestep" : self.max_timestep,
                "MinGain" : self.min_gain,
                "MaxGain" : self.max_gain,
                "MaxGenerateInitial" : self.max_generate_initial_population,
                "MaxGainValue" : self.max_gain_value,
                "SetPoint" : self.setpoint
            }

    def setProperties(self, prop):
        self.mutation_rate = prop["MutationRate"]
        self.crossover_rate = prop["CrossoverRate"]
        self.max_population = prop["MaxPopulation"]
        self.max_timestep = prop["MaxTimestep"]
        self.min_gain = prop["MinGain"]
        self.max_gain = prop["MaxGain"]
        self.max_generate_initial_population = prop["MaxGenerateInitial"]
        self.max_gain_value = prop["MaxGainValue"]
        self.setpoint = prop["SetPoint"]

    def setPopulation(self, popu):
        for i in range(len(popu)):
            self.population.append(i)

            self.population[i] = DNA(
                float(popu[i][0]),
                float(popu[i][1]),
                float(popu[i][2])
            )

            self.population[i].fitness = float(popu[i][3])
            self.population[i].risetime = float(popu[i][4])
            self.population[i].overshoot = float(popu[i][5])
            self.population[i].settling_time = float(popu[i][6])
            self.population[i].peak = float(popu[i][7])
            self.population[i].steadystate = float(popu[i][8])
            self.population[i].creator = popu[i][9]
            self.population[i].saved = popu[i][10]

    def generate_initial_population(self):
        #function for generate random individu in first iteration of evolution process
        random.seed()
        for i in range(self.max_generate_initial_population):
            self.population.append(i)
            self.population[i] = DNA(random.uniform(self.min_gain, self.max_gain),
                                        random.uniform(self.min_gain, self.max_gain*0.6),
                                        random.uniform(self.min_gain, self.max_gain*0.6))

    def add_to_population(self, individu):
        size_pop = len(self.population)
        self.population.append(size_pop+1)
        self.population[size_pop] = individu

    def pick_parent(self):
        #pick parent based on normalize probability
        random.seed()
        index = 0
        r = random.random()
        while(r > 0):
            r = r - self.population[index].normalize_prob
            index = index + 1
        index = index - 1
        return index

    def pick_best(self):
        i_best = np.amax(self.population, axis = 0)[3]
        for i in range(len(self.population)):
            if self.population[i][3] == i_best:
                break
        return self.population[i]

    def selection(self):
        sum_prob = 0
        parents = []

        s_population = len(self.population)

        print "\tCalculating Fitness..."
        for i in range(s_population):
            if self.population[i].fitness == 0:
                print "\t\tPopulation Index:",i
                self.population[i].calculate_fitness(self.max_timestep, self.setpoint, self.serial)

        for i in range(s_population):
            sum_prob = sum_prob + self.population[i].fitness
        for i in range(s_population):
            self.population[i].normalize_prob = self.population[i].fitness/float(sum_prob)

        if s_population >= self.max_population:
            self.max = True

        parentA = self.pick_parent()
        parentB = self.pick_parent()
        while parentA == parentB:
            parentB = self.pick_parent()
        parentA = self.population[parentA]
        parentB = self.population[parentB]
        parents = [parentA,parentB]

        return parents

    def mutation(self, parent):
        flag = False
        random.seed()
        if random.random() < self.mutation_rate:
            #adding old value with a very small random number
            parent.kp = parent.kp + random.random() / self.max_gain_value
            parent.ki = parent.ki + random.random() / self.max_gain_value
            parent.kd = parent.kd + random.random() / self.max_gain_value

            parent.fitness = 0

            #parent.creator = "mutation"

    def crossover(self,parentA, parentB):
        flag = False
        random.seed()
        if random.random() < self.crossover_rate:
            child = []
            for i in range(6):
                child.append(i)
            child[0] = DNA(parentA.kp,parentA.ki,parentB.kd)
            child[1] = DNA(parentA.kp,parentB.ki,parentB.kd)
            child[2] = DNA(parentA.kp,parentB.ki,parentA.kd)
            child[3] = DNA(parentB.kp,parentB.ki,parentA.kd)
            child[4] = DNA(parentB.kp,parentA.ki,parentA.kd)
            child[5] = DNA(parentB.kp,parentA.ki,parentB.kd)
            for i in range(6):
                for x in self.population:
                    #filter from duplicate DNA data
                    if x.kp == child[i].kp and x.ki == child[i].ki and x.kd == child[i].kd:
                        flag = True

                    #filter from data which kp < ki & kd
                    if child[i].kp < child[i].ki and child[i].kp < child[i].kd:
                        flag = True

                if flag == False:
                    self.add_to_population(child[i])
                    #child[i].creator = "crossover"
