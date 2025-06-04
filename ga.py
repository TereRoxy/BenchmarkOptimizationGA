import numpy as np
import random

class GeneticAlgorithm:
    def __init__(self, function, domain, encoding, crossover, pop_size, num_generations, mutation_rate, crossover_rate, seed):
        """
        Initialize Genetic Algorithm.
        Parameters:
            function: Function to optimize
            domain: Tuple of (min, max) for x and y
            encoding: 'binary' or 'real'
            crossover: '1-point', '2-point', 'arithmetic', or 'blx-alpha'
            pop_size: Population size
            num_generations: Number of generations
            mutation_rate: Probability of mutation (0-1)
            crossover_rate: Probability of crossover (0-1)
            seed: Random seed for reproducibility
        """
        self.function = function
        self.domain = domain
        self.encoding = encoding
        self.crossover = crossover
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.bit_length = 16
        random.seed(seed)
        np.random.seed(int(seed))
        self.population = self.initialize_population()

    def initialize_population(self):
        """
        Initialize the population based on the encoding type.
        :return: The initial population as a list of individuals.
        """
        if self.encoding == 'binary':
            return [random.choices([0, 1], k=self.bit_length*2) for _ in range(self.pop_size)]
        else:
            return [[random.uniform(self.domain[0], self.domain[1]),
                     random.uniform(self.domain[0], self.domain[1])] for _ in range(self.pop_size)]

    def binary_to_real(self, binary):
        x_bits, y_bits = binary[:self.bit_length], binary[self.bit_length:]
        x_int = int(''.join(map(str, x_bits)), 2)
        y_int = int(''.join(map(str, y_bits)), 2)
        max_int = 2**self.bit_length - 1
        x = self.domain[0] + (self.domain[1] - self.domain[0]) * x_int / max_int
        y = self.domain[0] + (self.domain[1] - self.domain[0]) * y_int / max_int
        return x, y

    def fitness(self, individual):
        """
        Calculate the fitness of an individual.
        """
        if self.encoding == 'binary':
            x, y = self.binary_to_real(individual)
        else:
            x, y = individual
        return -self.function(x, y)

    def select_parents(self):
        """
        Select two parents using tournament selection.
        """
        tournament_size = 3
        parents = []
        for _ in range(2):
            tournament = random.sample(self.population, tournament_size)
            best = max(tournament, key=self.fitness)
            parents.append(best)
        return parents

    def crossover_1point(self, parent1, parent2):
        """
        Perform 1-point crossover between two parents.
        :return: Tuple of two children.
        """
        if random.random() > self.crossover_rate:
            return parent1[:], parent2[:]
        point = random.randint(1, self.bit_length*2 - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    def crossover_2point(self, parent1, parent2):
        """
        Perform 2-point crossover between two parents.
        :return: Tuple of two children.
        """
        if random.random() > self.crossover_rate:
            return parent1[:], parent2[:]
        point1 = random.randint(1, self.bit_length*2 - 2)
        point2 = random.randint(point1 + 1, self.bit_length*2 - 1)
        child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
        child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
        return child1, child2

    def crossover_arithmetic(self, parent1, parent2):
        """
        Perform arithmetic crossover between two parents.
        :return: Tuple of two children.
        """
        if random.random() > self.crossover_rate:
            return parent1[:], parent2[:]
        alpha = random.random()
        child1 = [alpha * parent1[0] + (1 - alpha) * parent2[0], alpha * parent1[1] + (1 - alpha) * parent2[1]]
        child2 = [(1 - alpha) * parent1[0] + alpha * parent2[0], (1 - alpha) * parent1[1] + alpha * parent2[1]]
        return child1, child2

    def crossover_blx_alpha(self, parent1, parent2, alpha=0.5):
        """
        Perform BLX-alpha crossover between two parents.
        :param alpha: Alpha parameter for BLX-alpha crossover.
        :return: Tuple of two children.
        """
        if random.random() > self.crossover_rate:
            return parent1[:], parent2[:]
        child1, child2 = [], []
        for i in range(2):
            min_val = min(parent1[i], parent2[i])
            max_val = max(parent1[i], parent2[i])
            range_val = max_val - min_val
            child1.append(random.uniform(max(min_val - alpha * range_val, self.domain[0]), min(max_val + alpha * range_val, self.domain[1])))
            child2.append(random.uniform(max(min_val - alpha * range_val, self.domain[0]), min(max_val + alpha * range_val, self.domain[1])))
        return child1, child2

    def mutate(self, individual):
        """
        Mutate an individual based on the mutation rate.
        :param individual: The individual to mutate.
        :return: Mutated individual.
        """
        if self.encoding == 'binary':
            mutated = individual[:]
            for i in range(len(mutated)):
                if random.random() < self.mutation_rate:
                    mutated[i] = 1 - mutated[i]
            return mutated
        else:
            mutated = individual[:]
            for i in range(2):
                if random.random() < self.mutation_rate:
                    mutated[i] = random.uniform(self.domain[0], self.domain[1])
            return mutated

    def run(self):
        """
        Run the genetic algorithm for a specified number of generations.
        :return: Tuple of the best solution and its fitness.
        """
        best_solution = None
        best_fitness = float('-inf')
        for _ in range(self.num_generations):
            new_population = []
            for _ in range(self.pop_size // 2):
                parent1, parent2 = self.select_parents()
                if self.encoding == 'binary':
                    if self.crossover == '1-point':
                        child1, child2 = self.crossover_1point(parent1, parent2)
                    else:
                        child1, child2 = self.crossover_2point(parent1, parent2)
                else:
                    if self.crossover == 'arithmetic':
                        child1, child2 = self.crossover_arithmetic(parent1, parent2)
                    else:
                        child1, child2 = self.crossover_blx_alpha(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])

            self.population = new_population
            current_best = max(self.population, key=self.fitness)
            current_fitness = self.fitness(current_best)
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_solution = current_best

        if self.encoding == 'binary':
            best_solution = self.binary_to_real(best_solution)
        return best_solution, -best_fitness