import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.metrics import r2_score, mean_squared_error
from random import choices, randint

# process model
Kp = 3.0
taup = 5.0


def process(y, t, u, Kp, taup):
    # Kp = process gain
    # taup = process time constant
    dydt = -y / taup + Kp / taup * u
    return dydt


def closed_loop(Kp_, Ki, Kd):  # Simulates 1st order transfer function system with PID, returns MSE as fitness value
    # specify number of steps
    ns = 300

    # define time points
    t = np.linspace(0, ns / 10, ns + 1)
    delta_t = t[1] - t[0]

    # storage for recording values
    op = np.zeros(ns + 1)  # controller output
    pv = np.zeros(ns + 1)  # process variable
    e = np.zeros(ns + 1)  # error
    ie = np.zeros(ns + 1)  # integral of the error
    dpv = np.zeros(ns + 1)  # derivative of the pv
    P = np.zeros(ns + 1)  # proportional
    I = np.zeros(ns + 1)  # integral
    D = np.zeros(ns + 1)  # derivative
    sp = np.zeros(ns + 1)  # set point
    sp[25:] = 10

    # Upper and Lower limits on OP
    op_hi = 100.0
    op_lo = 0.0

    # loop through time steps
    for i in range(0, ns):
        e[i] = sp[i] - pv[i]
        if i >= 1:  # calculate starting on second cycle
            dpv[i] = (pv[i] - pv[i - 1]) / delta_t
            ie[i] = ie[i - 1] + e[i] * delta_t
        P[i] = Kp_ * e[i]
        I[i] = Ki * ie[i]
        D[i] = Kd * dpv[i]
        # I[i] = Kc / tauI * ie[i]
        # D[i] = - Kc * tauD * dpv[i]
        op[i] = op[0] + P[i] + I[i] + D[i]
        if op[i] > op_hi:  # check upper limit
            op[i] = op_hi
            ie[i] = ie[i] - e[i] * delta_t  # anti-reset windup
        if op[i] < op_lo:  # check lower limit
            op[i] = op_lo
            ie[i] = ie[i] - e[i] * delta_t  # anti-reset windup
        y = odeint(process, pv[i], [0, delta_t], args=(op[i], Kp, taup))
        pv[i + 1] = y[-1]
        op[ns] = op[ns - 1]
        ie[ns] = ie[ns - 1]
        P[ns] = P[ns - 1]
        I[ns] = I[ns - 1]
        D[ns] = D[ns - 1]
    MSE = mean_squared_error(sp, pv)
    r2 = r2_score(sp, pv)

    #print('MSE : %.2f' % MSE)
    # print('r2 : %.2f' % r2)
    return MSE, t, sp, pv


def generate_genome():
    return choices(population=np.linspace(0, 1000, 10000), k=3)


def generate_population(pop_size):
    population = []
    for i in range(pop_size):
        population.append(generate_genome())
    return population


def fitness_func(pop):
    fit_pop = []
    for i in range(len(pop)):
        #print(i)
        MSE_in, time, setptin, pv_ = closed_loop(pop[i][0], pop[i][1], pop[i][2])
        fit_pop.append(MSE_in)  # MSE is fitness function result. Lower is better
    return fit_pop


def selection_pair(pop, fit_pop):  # Selects two parents from population with fitness function
    return choices(
        population=pop,
        weights=fit_pop,
        k=2
    )


def fit_min(fit_pop, count):  # Returns list of top count number of index of results from generation
    fit_min_index = []
    rep_count = count
    temp_min = 13000
    last_i = -1
    for i in range(len(fit_pop)):
        if fit_pop[i] < temp_min:
            temp_min = fit_pop[i]
            last_i = i
    if last_i > -1:
        fit_min_index.append(last_i)

    def rec_min(in_fit_pop, minimum):
        nonlocal rep_count
        temp = 13000
        if rep_count > 0:
            last_j = -1
            for k in range(len(in_fit_pop)):
                if (in_fit_pop[k] < temp) and (in_fit_pop[k] > minimum):
                    temp = in_fit_pop[k]
                    last_j = k
            if last_j > -1:
                fit_min_index.append(last_j)
            rep_count = rep_count - 1
            rec_min(in_fit_pop, temp)
            return fit_min_index
        else:
            return fit_min_index

    output = rec_min(fit_pop, temp_min)
    return output


def fit_inv(fit_pop):  # Inverts fitness list values to make selection weight work correctly
    temp_max = 0
    for i in range(len(fit_pop)):
        if fit_pop[i] > temp_max:
            temp_max = fit_pop[i]
    print("Max gen MSE Val: " + str(temp_max))
    for i in range(len(fit_pop)):
        fit_pop[i] = temp_max - fit_pop[i]
    return fit_pop


def crossover(genome0, genome1):
    new_genome0 = []
    new_genome1 = []

    for x in range(3):
        p = randint(1, 11)
        mask = 0xFFF << p        # preserves upper part of genome
        mask_shift = 0xFFF >> p  # preserves lower part of genome
        genome0[x] = int(genome0[x] * 10)
        genome1[x] = int(genome0[x] * 10)
        a = (genome0[x] & mask) + (genome1[x] & mask_shift)
        a = a/10
        if a > 1000:
            a = 1000
        new_genome0.append(a/10)
        b = (genome0[x] & mask_shift) + (genome1[x] & mask)
        b = b/10
        if b > 1000:
            b = 1000
        new_genome1.append(b/10)

    return new_genome0, new_genome1

def mutation(pop):
    for n in range(3):
        for genome in range(len(pop)):
            prob = randint(0, 100)
            if prob < 11:
                p = randint(1, 11)
                pop[genome][n] = int(pop[genome][n] * 10)
                pop[genome][n] = pop[genome][n] ^ (1 << (p - 1))  # Toggle random bit
                pop[genome][n] = pop[genome][n] / 10
    return pop

def graph_op(t, sp, pv):
    plt.figure(1)
    plt.plot(t, sp, 'k-', linewidth=2)
    plt.plot(t, pv, 'b--', linewidth=3)
    plt.legend(['Set Point (SP)', 'Process Variable (PV)'], loc='best')
    plt.ylabel('Process')
    plt.ylim([-0.1, 12])
    plt.xlabel('Time')
    plt.show()


pop_init = generate_population(100)
print("Initial Population initiated")
fitness = fitness_func(pop_init) # fitness matrix with MSE
top_fitness_index = fit_min(fitness, 10)  # Index number of top 10 fitnesses

average = 0
for z in range(len(top_fitness_index)):
    average = average + fitness[top_fitness_index[z]]
    print("Index " + str(z) + " = " + str(fitness[top_fitness_index[z]]))

average = average / len(top_fitness_index)
print("Average of top 10 in generation = " + str(average))
print("Top fitness indexes are:" + str(top_fitness_index))

for m in range(100):
    print("=======================================================================")
    print("Gen " + str(m) + " Population initiated")

    next_gen = []

    # Elitism, picks top 6 genomes
    for elite in range(6):
        next_gen.append(pop_init[top_fitness_index[elite]])

    fitness = fit_inv(fitness)  # Matrix of fitnesses invert
    print("Top genome of generation = " + str(next_gen[0]))
    print("Pop_init length = " + str(len(pop_init)))

    for j in range(6, (len(pop_init)//2)+3):
        parent = selection_pair(pop_init, fitness)
        offspring = crossover(parent[0], parent[1])
        next_gen.append(offspring[0])
        next_gen.append(offspring[1])
    print("Gen length =" + str(len(next_gen)))

    mutation(next_gen)

    pop_init = next_gen

    fitness = fitness_func(pop_init)

    top_fitness_index = fit_min(fitness, 10)

    average = 0
    for z in range(len(top_fitness_index)):
        average = average + fitness[top_fitness_index[z]]
        print("Index " + str(z) + " = " + str(fitness[top_fitness_index[z]]))

    average = average / len(top_fitness_index)
    print("Average of top 10 in generation = " + str(average))

MSE_, t, sp, pv = closed_loop(pop_init[top_fitness_index[0]][0], pop_init[top_fitness_index[0]][1], pop_init[top_fitness_index[0]][2])
graph_op(t, sp, pv)