import numpy as np

class StepInfo:
    '''
    This class is for calculate and find step info from control data system
    step data require for calculate fitness function in genetic algorithm
    '''
    def __init__(self, x_list, y_list, setpoint):
        self.x_list = x_list
        self.y_list = y_list
        self.setpoint = setpoint
        self.rise_time_value = 0.9 * self.setpoint
        self.rise_time_index = 0
        self.error_band = 0.05 * self.setpoint
        self.low_band = self.setpoint - self.error_band 
        self.high_band = self.setpoint + self.error_band 
    
    def getRiseTime(self):
        #rise time is time for system from the beginning until system get 90% of setpoint
        self.rise_time_index = 0
        while self.rise_time_index<len(self.y_list):
            if self.y_list[self.rise_time_index] >= self.rise_time_value:
                break
            self.rise_time_index += 1
        return self.x_list[self.rise_time_index-1]
    
    def getPeak(self):
        #peak value is maximum y data of the system
        return np.amax(self.y_list)
    
    def getPeakTime(self):
        #peak time is time for system (or index) when system get 'peak' level
        l = np.where(self.y_list == np.amax(self.y_list))
        index = l[0][0]
        return self.x_list[index]

    def getOvershoot(self):
        #overshoot is percentage of 'peak' level according the last final data of y
        y_len = len(self.y_list) - 1
        last_y = self.y_list[y_len]
        return (self.getPeak() - self.setpoint)/(self.setpoint*1.0)*100

        '''
        y_len = len(self.y_list) - 1
        last_y = self.y_list[y_len]
        return (self.getPeak() - last_y)/(last_y*1.0)*100
        '''
    
    def getMSE(self):
        square_error = []
        for step in self.y_list:
            square_error.append(self.setpoint-step)
        
        square_error = np.square(square_error)

        return np.mean(square_error)

    
    def getSettlingTime(self):
        #settling time is range of time when system get atleast 5% of setpoint
        flag = []
        for i in range (len(self.y_list)):
            if self.y_list[i] >= self.low_band and self.y_list[i] <= self.high_band:
                flag.append(self.x_list[i])
            else:
                flag = []
        
        if not flag:
            return 0
        else:
            return flag[0] - self.x_list[0]
            
    def getSteadyStateError(self):
        #steady state error is value when system can't reach setpoint in infinite iteration
        p = len(self.y_list) -1
        return abs(self.y_list[p] - self.setpoint)