import random
import numpy as np
import matplotlib.pyplot as plt

class Chromosome:

    def __init__(self, a, b):
        self.a = a  
        self.b = b  

def initialize_population(population_size, a_range, b_range):

    population = []
    for _ in range(population_size):
        a = random.uniform(a_range[0], a_range[1])  
        b = random.uniform(b_range[0], b_range[1])  
        population.append(Chromosome(a, b))  
    return population

def calculate_mse(a, b, points):

    mse = 0
    for x, y in points:
        y_predicted = a * x + b  
        mse += (y - y_predicted)**2  
    return mse / len(points)

def fitness(a, b, points):

    mse = calculate_mse(a, b, points)  
    return 1 / (1 + mse)  

def roulette_wheel_selection(population, fitness_values):
    
    cumulative_probabilities = np.cumsum(fitness_values / np.sum(fitness_values))

    parent1_index = np.searchsorted(cumulative_probabilities, random.random())  
    parent2_index = np.searchsorted(cumulative_probabilities, random.random())  

    return population[parent1_index], population[parent2_index]  

def crossover(parent1, parent2, crossover_rate):
   
    if random.random() < crossover_rate:  
        child1_a = (parent1.a + parent2.a) / 2  
        child1_b = (parent1.b + parent2.b) / 2  
        child2_a = (parent1.a + parent2.a) / 2  
        child2_b = (parent1.b + parent2.b) / 2  
        return Chromosome(child1_a, child1_b), Chromosome(child2_a, child2_b)  
    else:
        return Chromosome(parent1.a, parent1.b), Chromosome(parent2.a, parent2.b) 

def mutate(chromosome, mutation_rate, mutation_scale, a_range, b_range):

    if random.random() < mutation_rate:  
        chromosome.a += random.gauss(0, mutation_scale)  
        chromosome.a = max(min(chromosome.a, a_range[1]), a_range[0]) 

    if random.random() < mutation_rate: 
        chromosome.b += random.gauss(0, mutation_scale)
        chromosome.b = max(min(chromosome.b, b_range[1]), b_range[0]) 
    return chromosome  

def genetic_algorithm(points, population_size=50, a_range=(-10, 10), b_range=(-100, 100),
                       crossover_rate=0.8, mutation_rate=0.05, mutation_scale=1.0,
                       max_generations=100, stagnation_threshold=10):


    population = initialize_population(population_size, a_range, b_range)  
    best_fitness = float('-inf')  
    stagnation_counter = 0
    best_chromosome = None  

    for generation in range(max_generations):  
        fitness_values = [fitness(chromosome.a, chromosome.b, points) for chromosome in population]

        current_best_fitness = max(fitness_values)  
        if current_best_fitness > best_fitness:  
            best_fitness = current_best_fitness
            stagnation_counter = 0 
            best_chromosome = population[fitness_values.index(current_best_fitness)]  
        else:
            stagnation_counter += 1  
            if stagnation_counter >= stagnation_threshold:  
                print(f"Остановка на {generation} поколении из-за отсутствия улучшений.")
                break  

        new_population = [] 
        for _ in range(population_size // 2):  
            parent1, parent2 = roulette_wheel_selection(population, fitness_values)  
            child1, child2 = crossover(parent1, parent2, crossover_rate)  
            child1 = mutate(child1, mutation_rate, mutation_scale, a_range, b_range)  
            child2 = mutate(child2, mutation_rate, mutation_scale, a_range, b_range)
            new_population.append(child1) 
            new_population.append(child2)

        population = new_population  

    if best_chromosome is None:
        fitness_values = [fitness(chromosome.a, chromosome.b, points) for chromosome in population]
        best_chromosome = population[fitness_values.index(max(fitness_values))]


    return best_chromosome.a, best_chromosome.b  

if __name__ == '__main__':
    
    points=[]
    for i in range(500):
        x = i
        y = 2 * x + 1 + random.gauss(50, 150)
        points.append((x, y))

    a, b = genetic_algorithm(points)

    print(f"Аппроксимирующая прямая: y = {a:.2f}x + {b:.2f}")

    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    plt.scatter(x_values, y_values, label="Точки данных")

    x_range = np.linspace(min(x_values), max(x_values), 100)
    y_predicted = a * x_range + b
    plt.plot(x_range, y_predicted, color='red', label="Аппроксимирующая прямая")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Линейная регрессия с помощью генетического алгоритма")
    plt.legend()
    plt.grid(True)
    plt.show()