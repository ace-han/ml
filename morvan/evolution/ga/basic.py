import numpy as np


DNA_SIZE = 10            # DNA length
# DNA_SIZE = 20            # DNA length [0, 5) divide into 2**DNA_SIZE
POP_SIZE = 100           # population size
CROSS_RATE = 0.8         # mating probability (DNA crossover)
MUTATION_RATE = 0.003    # mutation probability
N_GENERATIONS = 200
X_BOUND = [0, 5]         # x upper and lower bounds

def F(x): 
    # to draw a point on y-axis 
    return np.sin(10*x)*x + np.cos(2*x)*x     


def get_fitness(pred): 
    # find non-zero fitness for selection
    # avoid zero
    # return an np.array
    return pred + 1e-3 - np.min(pred)


def translateDNA(pop): 
    # convert binary DNA to decimal and normalize it to a range(0, 5)
    return pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2**DNA_SIZE-1) * X_BOUND[1]


def select(pop, fitness):    
    # nature selection wrt pop's fitness
    # the max fitness value get more chance to be selected
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=fitness/fitness.sum())
    return pop[idx]


def crossover(parent, pop):     
    # mating process (genes crossover)
    # crossover within the selected better fitness generation
    if np.random.rand() < CROSS_RATE:
        # select another individual from pop
        i_ = np.random.randint(0, POP_SIZE, size=1)                             
        # choose crossover points
        # it's total random along the DNA size array
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)   
        # mating and produce one child
        parent[cross_points] = pop[i_, cross_points]                            
    return parent


def mutate(child):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = 1 if child[point] == 0 else 0
    return child

if __name__ == '__main__':
    # print(translateDNA(np.array([1, 0 ,1, 0, 1, 0, 0, 0, 1, 0])))
    # exit(0)
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    # initialize the pop DNA
    pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE))   
    # something about plotting
    plt.ion()
    # 0...5, 200 equal units       
    x = np.linspace(*X_BOUND, 200)
    plt.plot(x, F(x))

    for _ in range(N_GENERATIONS):
        # compute function value by extracting DNA
        translated_x = translateDNA(pop)
        F_values = F(translated_x)    

        # something about plotting
        if 'sca' in globals(): 
            sca.remove()
        sca = plt.scatter(translated_x, F_values, s=100, lw=0, c='red', alpha=0.5)
        plt.pause(0.05)

        # GA part (evolution)

        # calculate fitness for each F_value
        fitness = get_fitness(F_values)
        # select the max index from fitness array
        best_child = pop[np.argmax(fitness), :]
        translated_x = translateDNA(np.array([best_child]))
        F_values = F(translated_x) 
        print("Most fitted DNA: ", best_child, translated_x, F_values)
        pop = select(pop, fitness)
        pop_copy = pop.copy()
        for parent in pop:
            child = crossover(parent, pop_copy)
            child = mutate(child)
            # parent is replaced by its child
            parent[:] = child       

    plt.ioff()
    plt.show()