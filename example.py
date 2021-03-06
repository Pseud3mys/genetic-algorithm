# import the algorithm class as ga
import genetic_algorithm as ga


# ----------#-----------------------------------------------------#---------- #
# ---------- a simple individual example for the genetic algorithm ---------- #

class simple_individual:
    def __init__(self, params=None, predefined_genome=None):
        import numpy as np

        # params is not used for this individual

        # the genome can be given by the genetic algorithm, or not
        self._genome = []  # private attribute, not used by the genetic algorithm
        if predefined_genome is None:
            # genome is not given, it can be generated by the individual
            self._genome = np.random.randint(-20, 20, 4)
        else:
            # genome is given by the argument "predefined_genome", it must not be an other thing
            self.set_genome(predefined_genome)

        # the individual need to save his how fitness (a float or an integer)
        self.fitness = 0  # inited at 0, it will be calculated by the "evaluate" function
        self.already_evaluate = False

    # return all the gene of an individual in a 1D array, (call genome).
    # (used as DNA by the genetic algorithm)
    def get_genome(self):
        #
        return self._genome  # the function must return the genome in a 1D array

    # take the genome in argument and modify the internal genome of the individual with it.
    # (used for mutation and reproduction by genetic algorithm)
    def set_genome(self, genome):
        self._genome = genome

    # function restraint is use to restraint the individual
    def restraint(self):
        summe = abs(self._genome[0]) + abs(self._genome[1]) + abs(self._genome[2]) + abs(self._genome[3])
        if summe > 40:
            return True

        # return False if it is not restraint,
        # OR you can return True to kill the individual
        # OR you can return a number (the absolute value will be subtracted from the fitness)
        return False

    # this function is use to determine the "fitness" of an individual
    # (used to sort and to reproduce good individual)
    def evaluate(self):
        terms = self._genome
        self.fitness = terms[0] * terms[1] + terms[2] - terms[3]
        # self.fitness = terms[0] * terms[1] * terms[2] * terms[3]

        if type(self.restraint()) is not bool:
            self.fitness += - abs(self.restraint())  # if it is a number it subtracted the fitness
        return self.fitness  # the function must return the calculated fitness


# settings
ga.setting_for_individual(ind_class=simple_individual, params=())

ga.setting_for_file(load="init_pop", save="nada")

ga.setting_for_selection(elitism=5, by_rank=8, by_fitness=8, by_tournament=13)

# if proba is 0.5 that mean that all selected individual have 0.5 to mutate
ga.setting_mutation(mutation_crossover=0, mutation_inverse=0, mutation_modify=0.5, modify_range=2, modify_number_type=float)

ga.set_mean_indiviudals_count(100)


# evolute on 3 diff ways:

# make x evolution to make a mean of the different value on the graph
# ga.multiple_evolutions(100, 15)

# evolute x generation:
# ga.evolute(100)

# evolute automatically with some param
ga.evolute(-1, auto_difference=0.01, auto_min_score=350)
