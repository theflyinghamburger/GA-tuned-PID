from random import choices, randint, randrange, random


def generate_genome():
    return choices(range(0.1, 100), k=3)


def generate_population(pop_size):
    population = []
    for i in range(pop_size):
        population.append(generate_genome())
    return population



print(generate_population(100))
