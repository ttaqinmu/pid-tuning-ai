import numpy as np
import time
import serial
import sys
import random
import csv
from ai_ga import DNA, Population
from ai_nn import NeuralNetwork
from file_model import FileModel


def ga(auto = False):
    if auto == False:
        i_m = raw_input("Open Genetic properties ?(y/n): ")
    else:
        i_m = "y"

    if i_m == "y":
        if auto == True:
            MUTATION_RATE = 0.3
            CROSSOVER_RATE = 0.7
            MAX_POPULATION = 1000
            MAX_TIMESTEP = 10000
            MAX_GAIN_VALUE = 2.5
            MIN_GAIN = 0
            MAX_GAIN = 10
            MAX_INIT_POPULATION = 5
            SETPOINT = 26.50
            pop = Population(
                MUTATION_RATE,
                CROSSOVER_RATE,
                MAX_POPULATION,
                MAX_TIMESTEP,
                MAX_GAIN_VALUE,
                MIN_GAIN,
                MAX_GAIN,
                MAX_INIT_POPULATION,
                SETPOINT,
                SERIAL
            )
            file_model.save_properties_to_model(pop.properties)
            print "Properties created"
        else:
            properties = file_model.open_properties_from_model()
            pop = Population(
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                SERIAL
            )
            pop.setProperties(properties)
            print "Properties Opened"
    else:
        print "Please input genetic properties"
        MUTATION_RATE = float(raw_input("Mutation rate (0~1): "))
        CROSSOVER_RATE = float(raw_input("Crossover rate (0~1): "))
        MAX_POPULATION = input("Maximal Population: ")
        MAX_TIMESTEP = input("Maximal Step: ")
        MAX_GAIN_VALUE = input("Maximal gain value: ")
        MIN_GAIN = input("Minimal gain random: ")
        MAX_GAIN = input("Maximal gain random: ")
        MAX_INIT_POPULATION = input("Maximal initial population: ")
        SETPOINT = input("Setpoint: ")

        pop = Population(
            MUTATION_RATE,
            CROSSOVER_RATE,
            MAX_POPULATION,
            MAX_TIMESTEP,
            MAX_GAIN_VALUE,
            MIN_GAIN,
            MAX_GAIN,
            MAX_INIT_POPULATION,
            SETPOINT,
            SERIAL
        )
        print "Properties created"

    if auto == False:
        i_m2 = raw_input("Open population model?(y/n): ")
    else:
        i_m2 = 'n'

    if i_m2 == 'y':
        popu = file_model.open_population_from_model()
        pop.setPopulation(popu)
        print "Population opened"
    else:
        print "AI > Genetics > Generating Initial Population..."
        pop.generate_initial_population()
        print "AI > Genetics > Generating random population done!"

    print "AI > Genetcis > Starting..."
    while True:
        print "AI > Genetics > Selection..."
        parent = pop.selection()

        print "AI > Genetics > Crossover..."
        pop.crossover(parent[0],parent[1])

        print "AI > Genetics > Mutation..."
        pop.mutation(parent[0])

        for i in range (len(pop.population)):
            if pop.population[i].saved == 0 and pop.population[i].fitness != 0:
                file_model.save_individu_to_model(pop.population[i])
                file_model.save_individu_to_graph(pop.population[i].x_step, pop.population[i].y_step, pop.setpoint, i)
                pop.population[i].saved = 1

        print "AI > Genetics > Population size: ",len(pop.population)

        if pop.max == True:
            print "AI > Genetics > Max Population reached!"
            print "AI > Genetics > Saving properties and population model..."
            file_model.save_population_to_model(pop.population)
            file_model.save_properties_to_model(pop.properties)
            print "AI > Genetics > Done!"
            break

def nn():
    nn = NeuralNetwork(5, 12, 3)
    data_ga = file_model.open_population_from_model()
    data_ga = nn.normalize(data_ga)
    len_ga = len(data_ga)
    dataset = [{}]

    print "AI > Neural > Creating dataset from Genetic population..."

    for i in range(len_ga-1):
        dataset.append({
            "output" : [float(data_ga[i][0]),float(data_ga[i][1]),float(data_ga[i][2])],
            "input" : [float(data_ga[i][4]),float(data_ga[i][5]),float(data_ga[i][6]),float(data_ga[i][7]),float(data_ga[i][8])]
        })

    random.seed()
    print "AI > Neural > Preparing for training with default iteration (1000)..."
    for i in range(100000):
        print "AI > Neural > Training iteration",i+1
        index = int(random.uniform(0,len_ga))
        if index > len_ga-1:
            index = len_ga-1
        if index == 0:
            index = 1
        nn.train(dataset[index]["input"], np.matrix(dataset[index]["output"]))

    print "AI > Neural > Training Done!"

    print "AI > Neural > Predicting the best tuning for PID..."
    risetime = 25
    overshoot = 5
    settling = 50
    peak = 26
    steady = 0

    output = nn.predict([risetime,overshoot,settling,peak,steady])
    print ""
    print "\tKP:",output[0,0]," KI:",output[0,1]," KD:",output[0,2]
    print ""
    print "AI > Neural > Done!"

    d = raw_input("AI > Neural > Save Weight and Bias? (y/n):")
    if d == 'y':
        nn.save()

def nn_predict():
    c = 'y'

    nn = NeuralNetwork(5,12,3)

    while c == 'y':
        c = raw_input("AI > Neural > Try prediction? (y/n):")
        if c == 'n':
            break

        risetime = input("rise time:")
        overshoot = input("overshoot:")
        settling = input("settling time:")
        peak = input("peak:")
        steady = input("steady-state error:")
        output = nn.predict_from_model([risetime,overshoot,settling,peak,steady])

        print ""
        print "\tKP:",output[0,0]," KI:",output[0,1]," KD:",output[0,2]
        print ""
        print "AI > Neural > Done!"

def keras():
    from keras.models import model_from_json
    import numpy as np

    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='mse', optimizer='rmsprop')

    risetime = 25
    overshoot = 1
    settling = 50
    peak = 26
    steady = 0.0001

    predict = np.array([[risetime],[overshoot],[settling],[peak],[steady]])
    predict = np.transpose(predict)

    print(loaded_model.predict(predict))
    '''
    from keras.models import Sequential, model_from_json
    from keras.layers.core import Dense, Dropout, Activation
    from keras.optimizers import SGD

    data_ga = [[]]
    fol = "tes.csv"
    with open(fol) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            data_ga.append(1)
            data_ga[line_count] = [
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

    len_ga = len(data_ga)

    input = []
    output = []

    for i in range(len_ga-1):
        input.append((float(data_ga[i][4]),float(data_ga[i][5]),float(data_ga[i][6]),float(data_ga[i][7]),float(data_ga[i][8])))
        output.append(([float(data_ga[i][0]),float(data_ga[i][1]),float(data_ga[i][2])]))


    x = np.array(input)
    y = np.array(output)
    model = Sequential()

    model.add(Dense(45,activation='relu'))
    model.add(Dense(30,activation='relu'))
    model.add(Dense(12,activation='relu'))
    model.add(Dense(3))

    sgd = SGD(lr=0.1)
    model.compile(loss='mse', optimizer='rmsprop')

    model.fit(x, y, batch_size=50, epochs=1000)

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")

    risetime = 29
    overshoot = 3.34052031556013
    settling = 30
    peak = 25.8346100019
    steady = 0.000503265310076
    #1.83398628969672,1.58367958575964,0.309875038875906

    predict = np.array([[risetime],[overshoot],[settling],[peak],[steady]])
    predict = np.transpose(predict)

    print(model.predict_proba(predict))
    '''




SERIAL = serial.Serial("/dev/ttyACM0", 115200)

print "Opening Serial..."
#time.sleep(2)
print "Done!"

while True:
    i_c = raw_input("Calibrating ESC ?(y/n): ")
    if i_c == 'y':
        SERIAL.write('c')
        time.sleep(5)
        print "Calibrating done (sending 1000ms signal to ESC)!"
    elif i_c == 'n':
        break
    else:
        print "Input must 'y' (yes) or 'n' (no)!"

print "\t\t\t\t-----------------------------------------------"
print "\t\t\t\t PID Tuning with Artificial Intelegence method"
print "\t\t\t\t             Tower Copter Control"
print "\t\t\t\t               M.Imam Muttaqin"
print "\t\t\t\t-----------------------------------------------"
print "\t\t\t\t Select Mode > 1.Automatic (use default-conf)"
print "\t\t\t\t             > 2.Genetic Algorithm (Hard Tune)"
print "\t\t\t\t             > 3.Neural Network (Soft Tune)"
print "\t\t\t\t             > 4.TensorFlow (Soft Tune)"
print "\t\t\t\t             > 5.NN Predict"
print "\t\t\t\t             > 99.Exit"

i_mode = input("\t\t\t\t             : ")

name_model = raw_input("Enter name of models: ")
file_model = FileModel(name_model)

if i_mode == 1:
    ga(True)
    nn()
elif i_mode == 2:
    ga()
elif i_mode == 3:
    nn()
elif i_mode == 4:
    keras()
elif i_mode == 5:
    nn_predict()
else:
    sys.exit()
