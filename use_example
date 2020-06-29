# import the algorithm with tools and individuals_examples
from geneticAlgorithm import algorithm, tools, individuals_examples


# first the parameters for the individuals:
params_individuals = algorithm.default_params_individuals  # load the default value
# initial population size, less the individuals illegal.. (if not loaded from a file)
params_individuals[0] = 30
# mean size for the population
params_individuals[1] = 50
# the class for all individuals, see example of this in "individuals_examples.py"
params_individuals[2] = individuals_examples.simple_individual
# and optionally some argument for the class in a tuple
params_individuals[3] = (None, None)


# now the parameters for the generation of the population:
params_probas = algorithm.default_params_probability   # load the default value

# were we leave the default value and just comment the meaning.
# all of the number are probability.
params_probas[0] = [0, True]  # probability for elitism, force minimum 1 elite (True or False)
params_probas[1] = 0.1  # selection by rank.
params_probas[2] = 0.05  # selection by fitness.
params_probas[3] = 0.05  # selection by tournament.
params_probas[4] = 0.08  # mutation.
params_probas[5] = 0.05  # mutation multi points.

# and finally we create the genetic algorithm class:
example = algorithm.genetic_algorithm(params_individuals, params_probas, save_folder=None)

input("press any key to start evolution.\n")
# the tools functions allow you to evolute the algorithm automatically and show a graph at the end.
# save_file=None if you don't want to save the population
tools.genetic_algo_evolute(example, 50, load_file=None, save_file="file_name.npy")

input("\npress any key to start average evaluation.")
# or to generate a lots of the same generation to evaluate the parameters with a nice graph.
tools.genetic_algo_test_params(example, load_file="file_name.npy", number_generation_by_test=50, number_of_test=10)
