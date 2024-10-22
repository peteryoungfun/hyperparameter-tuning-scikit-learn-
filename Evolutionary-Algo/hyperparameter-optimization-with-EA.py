import numpy as np
import pandas as pd
from random import choice
import copy


# a function to initialize the evolutionary algorithm
# the inputs are three kinds of parameters and the number of random entities in the first generation
# the outputs are the names and information of the parameters and the first generation

def initialize(continuous_list, int_list, label_list, nums):
    parameter_info = dict()
    parameter_names = []
    initialization = pd.DataFrame()
    if len(continuous_list) != 0:
        for parameter in continuous_list:
            parameter_info[parameter[0]] = ('continuous', parameter[1])
            parameter_names.append(parameter[0])
            initialization[parameter[0]] = np.random.uniform(parameter[1][0], parameter[1][1], nums)
    if len(int_list) != 0:
        for parameter in int_list:
            parameter_info[parameter[0]] = ('int', parameter[1])
            parameter_names.append(parameter[0])
            initialization[parameter[0]] = [parameter[1][i] for i in np.random.randint(0, len(parameter[1]), nums)]
    if len(label_list) != 0:
        for parameter in label_list:
            parameter_info[parameter[0]] = ('label', parameter[1])
            parameter_names.append(parameter[0])
            initialization[parameter[0]] = [parameter[1][i] for i in np.random.randint(0, len(parameter[1]), nums)]
    return (parameter_names, parameter_info, initialization)

# a function to generate a child from two parents
# the inputs are two parents, parameter information and mutation chance
# the output is one child

def offspring(line1, line2, parameter_names, parameter_info, chance):
    child = pd.DataFrame()
    prob = float(line1.iloc[0:]['performance']) / (float(line1.iloc[0:]['performance']) + float(line2.iloc[0:]['performance']))

    for parameter in parameter_names:
        if parameter_info[parameter][0] == 'continuous':
            if np.random.random() > chance:
                if np.random.random() > prob:
                    child[parameter] = [float(line2.iloc[0:][parameter])]
                else:
                    child[parameter] = [float(line1.iloc[0:][parameter])]
            else:
                child[parameter] = [float(np.random.uniform(parameter_info[parameter][1][0], parameter_info[parameter][1][1], 1))]
        elif parameter_info[parameter][0] == 'int':
            if np.random.random() > chance:
                if np.random.random() > prob:
                    child[parameter] = [float(line2.iloc[0:][parameter])]
                else:
                    child[parameter] = [float(line1.iloc[0:][parameter])]
            else:
                child[parameter] = [choice(parameter_info[parameter][1])]
        else:
            if np.random.random() > chance:
                if np.random.random() > prob:
                    child[parameter] = [line2.iloc[0, :][parameter]]
                else:
                    child[parameter] = [line1.iloc[0, :][parameter]]
            else:
                child[parameter] = [choice(parameter_info[parameter][1])]
    return child

# a function to generate a new generation based on the data from parents
# for imput details, check the main description for evolution_parameter_optimization
# the output is a DataFrame of new generation

def breed(target_function, performance_function, data, parameter_names, order, parameter_info, settings, log):
    parents = copy.deepcopy(data)
    parents = parents.iloc[0 : int(settings[0] * settings[2])]

    total_performance = parents['performance'].sum()
    parents['prob'] = parents['performance'] / total_performance
    parents['cum_prob'] = parents['prob'].cumsum(axis = 0)
    new_data = pd.DataFrame(columns = parameter_names)

    for _ in range(settings[0]):
        line1 = parents[parents['cum_prob'] > np.random.random()].iloc[0:1]
        line2 = parents[parents['cum_prob'] > np.random.random()].iloc[0:1]
        child = offspring(line1, line2, parameter_names, parameter_info, settings[3])
        new_data = pd.concat([new_data, child], ignore_index = True)

    new_data['result'] = new_data[order].apply(lambda x: target_function(*x), axis = 1)
    new_data['performance'] = performance_function(new_data['result'])

    if settings[1] == True:
        new_data = pd.concat([new_data, data.iloc[0 : int(settings[0] * settings[2])]], ignore_index = True)
        new_data.sort_values(by = ['performance'], inplace = True, ascending = False)
        new_data = new_data.iloc[0:settings[0]]
    else:
        new_data.sort_values(by = ['performance'], inplace = True, ascending = False)

    log.append(float(new_data.iloc[0:1]['result']))

    return new_data

# define a default performance calculation
# the default measurement for performance is the squared result

def default_performance(x):
    return x ** 2

'''
The main function
    Target_Function is the function in which we want to optimize its parameters
        for example, f(x, y, z) = x * y + z with x, y, z being the parameters
    
    Performance_Function is the function to transfer results to performances of the parameters
        The default performance function is default_performance = result ** 2
    
    Parameters are the parameters of the target function
        the input is a list of three ORDERED kinds of parameters: continuous_parameters, int_parameters, label_parameters
            for example, target function f(c1, c2, i1, i2, l1, l2) has six parameters where:
                c1 is continuous and is in [100, 500]
                c2 is continuous and is in (0, 1)
                i1 is discrete and is in [1, 2, 3, 4, 5]
                i2 is discrete and is in [0.2, 0.4, 0.6, 0.8]
                l1 is a label and is in ['Male', 'Female']
                l2 is a label and is in ['x', 'y', 'z']
            then the input should be [ [['c1', [100, 500]], ['c2', [0, 1]]], [['i1', [1, 2, 3, 4, 5]], ['i2', [0.2, 0.4, 0.6, 0.8]]], [['l1', ['Male', 'Female']], ['l2', ['x', 'y', 'z']]] ]
            to be clear, we can also let
                continuous_list = [ ['c1', [100, 500]], ['c2', [0, 1]] ]
                int_list = [ ['i1', [1, 2, 3, 4, 5]], ['i2', [0.2, 0.4, 0.6, 0.8]] ]
                label_list = [ ['l1', ['Male', 'Female']], ['l2', ['x', 'y', 'z']] ]
            and the input is [continuous_list, int_list, label_list]
            if a target function only has two or less kinds of parameters, just put an empty list in the missing parameter.
            For example, [continuous_list, [], label_list]
    
    Order is the order of all parameters in the DEFINITION of the target function
        for example, for target function f(l1, c2, c1, i1, l2, i2), the order is ['l1', 'c2', 'c1', 'i1', 'l2', 'i2']
    Settings is the setting for the evolutionary algorithm, the default is (50, True, 0.2, 0.1)
        the first one (50) is the number of entities in each generation, the more entities the faster the evolution is, but requires more computing power
        the second one (True) is whether the algorithm will include parents in the next generation, set True will be faster but sensitive to noise
        the third one (0.2) is the ratio of entities that can be a parent, in this case only the top 50*0.2=10 entities can have child
        the fourth one (0.1), is the mutation chance, the bigger the algorithm is better at avoiding local maximum but is slower
    End decides how the algorithm ends
        ('iteration', 100) is default and means the algorithm will end after a certain amount of iterations (100 by default)
            the more iterations, the better but requires more computing power
        ('accuracy', 0.001) means the algorithm will end if the improvement of one iteration is less than the set accuracy
            is 'accuracy' is chosen, the second element of Settings must be False
            for parameter optimization, 'accuracy' is NOT RECOMMENDED, it is much faster but sometimes it will end far from the optimum
'''

def evolution_parameter_optimization(target_function, parameters, order, performance_function = default_performance, settings = (50, True, 0.2, 0.1), end = ('iteration', 100)):
    initialization = initialize(parameters[0], parameters[1], parameters[2], settings[0])
    parameter_names = initialization[0]
    parameter_info = initialization[1]
    generation = initialization[2]
    log = [0]
    del initialization

    generation['result'] = generation[order].apply(lambda x: target_function(*x), axis = 1)
    generation['performance'] = performance_function(generation['result'])
    generation.sort_values(by = ['performance'], inplace = True, ascending = False)
    log.append(float(generation[0:1]['result']))

    if end[0] == 'iteration':
        for _ in range(end[1]):
            generation = breed(target_function, performance_function, generation, parameter_names, order, parameter_info, settings, log)
    else:
        while abs(log[-1] - log[-2]) > end[1]:
            generation = breed(target_function, performance_function, generation, parameter_names, order, parameter_info, settings, log)

    return generation.iloc[0:1], log

# to test and better understand the functions
if __name__ == '__main__':

    # define a test function
    # the test function simulation the score of a data scientist graduate in the job market for a medium company IN CHINA
    # 'grade' is a float between 2.00 and 4.00, around 3.5 is prefered (too high may be fake)
    # 'background' is a float between 0.00 and 1.00 which measures the degree of professional fit
    # 'age' is an int between 18 and 34, 26-30 is prefered
    # 'internship' is the number of interships that can be [0, 1, 2, 3, 4, 5]
    # 'school' is a label in ['top4', '985', '211', 'level-1', 'level-2', 'level-3', 'others']
    # 'level' is a label in ['professor', 'doctor', 'master', 'bachelor', 'others']

    def test(grade, background, age, internship, school, level):
        school_score = {'top4': 1.5, '985': 1.2, '211': 1.0, 'level-1': 0.8, 'level-2': 0.6, 'level-3': 0.4, 'others': 0.2}
        level_score = {'professor': 0.8, 'doctor': 1.0, 'master': 1.2, 'bachelor': 1.0, 'others': 0.5}
        internship_score = {0:0.5, 1:1, 2:1.2, 3:1.5, 4:1, 5:0.8}
        age_score = -(age - 28) ** 2 + 100
        grade_score = (-(grade - 3.5) ** 2 + 2.25) * 4 / 9
        return grade_score * background * age_score * internship_score[internship] * school_score[school] * level_score[level] / 2

    print('top optimum score is: ', test(3.5, 1.00, 28, 3, 'top4', 'master'))

    continuous_list = [ ['grade', [2.00, 4.00]], ['background', [0.00, 1.00]] ]
    int_list = [ ['age', range(18, 35, 1)], ['internship', range(0, 6, 1)] ]
    label_list = [ ['school', ['top4', '985', '211', 'level-1', 'level-2', 'level-3', 'others']], ['level', ['professor', 'doctor', 'master', 'bachelor', 'others']] ]
    order = ['grade', 'background', 'age', 'internship', 'school', 'level']
    parameters = [continuous_list, int_list, label_list]
    settings = (50, True, 0.2, 0.1)
    end = ('iteration', 100)

    res = evolution_parameter_optimization(test, parameters, order, performance_function = default_performance, settings = settings, end = end)

    import matplotlib.pyplot as plt

    plt.plot(res[1])
    plt.show()

    res[0]
