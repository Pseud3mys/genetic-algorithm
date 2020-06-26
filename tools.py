
def print_population(pop):
    i = 0
    print("old population: ")
    for ind in pop:
        print(" - rank " + str(i) + ": " + get_individual_infos(ind))
        i += 1


def get_individual_infos(individual):
    if individual.already_evaluate:
        fitness = str(individual.fitness)
    else:
        fitness = "(not evaluate)"
    genome = individual.get_genome()
    if len(genome) > 10:
        genome = str(genome[0:10]) + "..., len: " + str(len(genome)) + "."
    else:
        genome = str(genome) + ", len=" + str(len(genome)) + "."
    return " genome: " + genome + " Fitness: " + fitness


def genetic_algo_test_params(algorithm_object, load_file, number_generation_by_test=50, number_test=10):
    all_max = []
    all_min = []
    all_median = []
    all_pop_len = []

    # process the genetics algorithms:
    print("starting evolution...")
    for i in range(number_test):
        all_max.append([])
        all_min.append([])
        all_median.append([])
        all_pop_len.append([])
        if load_file is None:
            algorithm_object.create_random_old_population(debug=False)
        else:
            algorithm_object.load_old_population(load_file, debug=False)
        for number_gen in range(number_generation_by_test):
            algorithm_object.create_next_generation()
            max_val, median, min_val = algorithm_object.get_old_pop_infos()
            all_max[i].append(max_val)
            all_min[i].append(min_val)
            all_median[i].append(median)
            all_pop_len[i].append(len(algorithm_object.old_population))
        print("    finished evolution of " + str(i) + "/" + str(number_test))

    # for the graph now:
    x = np.linspace(0, number_generation_by_test, number_generation_by_test)
    max_fitness = []
    min_fitness = []
    median = []
    for gen in range(number_test):
        max_fitness = np.mean(all_max, axis=0)
        min_fitness = np.mean(all_min, axis=0)
        median = np.mean(all_median, axis=0)
        plt.plot(x, all_max[gen], color="green", linewidth=2, linestyle="-", alpha=0.08)

    # find better evolution
    _max = -9999
    better_index = 0
    for index in range(len(all_max)):
        if all_max[index][len(all_max[index]) - 1] > _max:
            _max = all_max[index][len(all_max[index]) - 1]
            better_index = index

    # print infos
    print("\nat the end of the evolution:")
    print(" - better fitness: " + str(all_max[better_index][len(all_max[0]) - 1]))
    print(" - average fitness: " + str(max_fitness[len(max_fitness) - 1]))
    print(" - average median:  " + str(median[len(median) - 1]))

    # show graph
    print("opening graph...")
    plt.plot(x, all_max[better_index], color="green", linewidth=2, linestyle="-", alpha=0.4)
    plt.plot(x, all_median[better_index], color="blue", linewidth=2, linestyle="-", alpha=0.35)
    plt.plot(x, max_fitness, color="green", linewidth=2, linestyle="-")
    plt.plot(x, median, color="blue", linewidth=2, linestyle="-")
    plt.ylabel("fitness")
    plt.xlabel("number of generations")
    plt.fill_between(x, min_fitness, max_fitness, color="gray", alpha=0.3)
    plt.show()


def genetic_algo_evolute(algorithm_object, number_of_generation, load_file=None, save_file=None):
    if load_file is None:
        algorithm_object.create_random_old_population()
    else:
        algorithm_object.load_old_population(file_name=load_file)
    print_population(algorithm_object.old_population)

    print("start population info: ")
    max_val, median, min_val = algorithm_object.get_old_pop_infos()
    print(" - best individual: " + get_individual_infos(algorithm_object.old_population[0]))
    print(" - fitness median: " + str(median))

    max_array = []
    min_array = []
    median_array = []
    len_pop_array = []

    # create x generation
    for i in range(number_of_generation):
        algorithm_object.create_next_generation()
        max_val, median, min_val = algorithm_object.get_old_pop_infos()
        max_array.append(max_val)
        median_array.append(median)
        min_array.append(min_val)
        len_pop_array.append(len(algorithm_object.old_population))

    if save_file is not None:
        algorithm_object.save_old_population(save_file)

    print("final population info: ")
    max_val, median, min_val = algorithm_object.get_old_pop_infos()
    print(" - best individual: " + get_individual_infos(algorithm_object.old_population[0]))
    print(" - fitness median: " + str(median))

    x = np.linspace(0, number_of_generation, number_of_generation)
    plt.plot(x, len_pop_array, color="yellow", linewidth=2, linestyle="-", label="population size")
    plt.plot(x, max_array, color="green", linewidth=2, linestyle="-", label="max fitness")
    plt.plot(x, median_array, color="blue", linewidth=2, linestyle="-", label="median fitness")
    plt.ylabel("fitness")
    plt.xlabel("number of generations")
    plt.fill_between(x, min_array, max_array, color="gray", alpha=0.3)
    plt.legend()
    plt.show()
