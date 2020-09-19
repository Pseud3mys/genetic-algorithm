"""
a genetic algorithm...
"""
import os
from tqdm import tqdm

# other import
import matplotlib.pyplot as plt
import numpy as np


# global variables:
_individual_setting = [classmethod, None]
_file_setting = (None, None)
# selection:
_elitism = _by_rank = _by_fitness = _by_tournament = None
# mutation
_mutation_crossover = _mutation_inverse = _mutation_modify = _modify_range = _modify_number_type = None
_individuals_count = 0

# directory to load and save file (init with setting_for_file)
_save_dir = ""


def setting_for_individual(ind_class, params=(None, None)):
    global _individual_setting
    _individual_setting = [ind_class, params]


def setting_for_file(load=None, save=None):
    global _file_setting, _save_dir
    _save_dir = os.getcwd() + "\\"
    if load is not None:
        l = load.split(sep=".")
        load = l[0] + ".npy"
    if save is not None:
        s = save.split(sep=".")
        save = s[0] + ".npy"
    _file_setting = (load, save)


def setting_for_selection(elitism=5, by_rank=8, by_fitness=8, by_tournament=13):
    global _elitism, _by_rank, _by_fitness, _by_tournament
    _elitism = elitism
    _by_rank = by_rank
    _by_fitness = by_fitness
    _by_tournament = by_tournament


def setting_mutation(mutation_crossover, mutation_inverse, mutation_modify, modify_range=1, modify_number_type=float):
    global _mutation_crossover, _mutation_inverse, _mutation_modify, _modify_range, _modify_number_type
    _mutation_crossover = mutation_crossover
    _mutation_inverse = mutation_inverse
    _mutation_modify = mutation_modify
    _modify_range = modify_range
    _modify_number_type = modify_number_type


def set_mean_indiviudals_count(count):
    global _individuals_count
    _individuals_count = count


class genetic_algorithm:
    def __init__(self):
        global _individual_setting, _file_setting

        # individual params:
        self._default_load_file, self._default_save_file = _file_setting

        self._individuals_class = _individual_setting[0]
        self._individuals_class_param = _individual_setting[1]

        global _individuals_count
        self._mean_individuals_count = _individuals_count

        # print some infos/debug:
        print("\n-------- genetic algorithm infos --------")
        # create a fake individual for the debug
        alfred = self._individuals_class(self._individuals_class_param, None)
        print("\n - individuals:")
        print("          - number by generation: ~" + str(self._mean_individuals_count))
        self._len_genome = len(alfred.get_genome())
        print("          - number of genes: " + str(self._len_genome))
        print("          - class: " + str(self._individuals_class))
        print("\n - generations:")
        print("          - number of elite: " + str(_elitism))
        print("          - number of selection by rank: " + str(_by_rank))
        print("          - number of selection by fitness: " + str(_by_fitness))
        print("          - number of selection tournament: " + str(_by_tournament))
        total = _elitism + _by_rank + _by_fitness + _by_tournament
        print("          - Total of selected individuals: %d/%d" % (total, self._mean_individuals_count))
        mean = total*-1
        print("          - mean mutations by generation: %d" % 0)
        print("------------------------------------------\n\n")
        # create the pop
        self.old_population = np.array(([]))
        self.population = np.array(([]))

    def _add_new_individual_to_NEWpopulation(self, genome):
        new_individual = self._individuals_class(self._individuals_class_param, genome)
        self.population = np.append(self.population, new_individual)

    """def _predict_mutation_count(self):
        global _mutation_mode, _MODE_inverse, _MODE_modify, _MODE_restraint
        if self._mutation_mode == _MODE_restraint:
            prob, = self._mutation_params
        elif self._mutation_mode == _MODE_inverse:
            prob, = self._mutation_params
        elif self._mutation_mode == _MODE_modify:
            prob, = self._mutation_params
            self.inverse_mutation(self._mutation_params)"""

    # ---------- functions for the save and first population ---------- #

    def create_random_old_population(self, count_of_individual, debug=True):
        # on reset les pop:
        self.population = np.array(([]))
        self.old_population = np.array(([]))
        for i in range(count_of_individual):
            # create new people with a random genome
            new_individu = self._individuals_class(self._individuals_class_param, None)
            self.old_population = np.append(self.old_population, new_individu)
        # finally we rank the pop randomly generated
        self.rank_population()
        if debug:
            print("SAVED " + str(len(self.old_population)) + " individus with " + str(self._len_genome) + " genes each.")

    def load_old_population(self, file_name, debug=True):
        global _save_dir
        # on reset les pop:
        self.population = np.array(([]))
        self.old_population = np.array(([]))
        genomes_array = np.load(_save_dir + file_name)
        for genome in genomes_array:
            # create new people with a predefine genome
            new_individu = self._individuals_class(self._individuals_class_param, genome)
            self.old_population = np.append(self.old_population, new_individu)
        # finally we rank the pop loaded
        self.rank_population()
        if debug:
            print("LOADED %d individuals with %d gene each from \"%s\"" % (len(genomes_array), len(genomes_array[0]), _save_dir + file_name))

    def save_old_population(self, file_name, debug=True):
        global _save_dir
        genomes_array = []
        for ind in self.old_population:
            genomes_array.append(ind.get_genome())

        np.save(_save_dir + file_name, genomes_array)
        if debug:
            print("LOADED %d individuals with %d gene each in \"%s\"" % (len(genomes_array), len(genomes_array[0]), _save_dir + file_name))

    # ---------- functions for create the next generation ---------- #

    def filter_population(self):
        to_del = []
        for ind in range(len(self.old_population)):
            need_restraint = self.old_population[ind].restraint()
            if need_restraint and type(need_restraint) == bool:
                # print("kill a illegal ind: "+self._get_individual_info(self.old_population[ind]))
                to_del.append(ind)
        # puis on les supprimes
        self.old_population = np.delete(self.old_population, to_del)

    def rank_population(self):
        self.filter_population()  # filter before to speed the process
        # sort the population from the best fitness to the worst.
        i = 0
        to_evaluate = []
        for ind in self.old_population:
            if not ind.already_evaluate:
                i += 1
                to_evaluate.append(ind)
                ind.evaluate()
                ind.already_evaluate = True

        # Parallel(n_jobs=11)(delayed(self.evaluate_individual)(i) for i in to_evaluate)


        # les trie en fonction de leur fitness
        self.old_population = sorted(self.old_population, key=lambda x: x.fitness, reverse=True)

    # crossover between a exiting individual and a new random individual
    def crossover_mutation(self, probability):
        real_proba = probability/self._len_genome
        count = len(self.population)
        for i in range(count):
            random_ind = self._individuals_class(self._individuals_class_param, None)  # new random individual
            individu = np.random.choice(self.population)
            ADN = individu.get_genome()
            mutation = random_ind.get_genome()

            proba_array = np.random.random(self._len_genome)
            array, = np.where(proba_array <= real_proba)
            if array.size == 0:
                continue
            # range = 0.5: [-0.5, +0.5]
            ADN = np.where(proba_array <= real_proba, mutation, ADN)
            # set the muted genome
            self._add_new_individual_to_NEWpopulation(ADN)

    # metode de mutation qui inverse simple 2 genes
    def inverse_mutation(self, probability):
        count = len(self.population)
        for i in range(count):
            # test la proba
            if np.random.random() > probability:
                continue

            individu = np.random.choice(self.population)
            ADN = individu.get_genome()

            # index des 2 points à croiser:
            index = [i for i in range(len(ADN))]
            p = np.random.choice(index, 2, replace=False)

            gene = ADN[p[0]]
            ADN[p[0]] = ADN[p[1]]
            ADN[p[1]] = gene

            self._add_new_individual_to_NEWpopulation(ADN)

    # modify a little bit the value of some genes
    def modify_mutation(self, probability, modify_range, number_type):
        real_proba = probability/self._len_genome
        number_individuals = len(self.population)
        for i in range(number_individuals):
            individu = np.random.choice(self.population)
            ADN = individu.get_genome()

            proba_array = np.random.random(self._len_genome)
            array, = np.where(proba_array <= real_proba)
            if array.size == 0:
                continue
            # range = 0.5: [-0.5, +0.5]
            if number_type == float:
                ADN = np.where(proba_array <= real_proba, ADN + (2 * modify_range * (np.random.random() - 0.5)), ADN)
            elif number_type == int:
                # add just an int not a float
                ADN = np.where(proba_array <= real_proba, ADN + int(2 * modify_range * (np.random.random() - 0.5)), ADN)
            elif number_type == bool:
                # if it is 0 became 1 and vice versa
                ADN = np.where(proba_array <= real_proba, -ADN+1, ADN)
            else:
                raise
            # set the muted genome
            self._add_new_individual_to_NEWpopulation(ADN)

    def crossover(self, number_of_crossing_point=-1):
        count = (self._mean_individuals_count - len(self.population)) / 2
        for _ in range(int(count)):
            # croise les individus deja selectionné pour le génération suivant entre eux
            pere = np.random.choice(self.population)
            mere = np.random.choice(self.population)
            genome_pere = pere.get_genome()
            genome_mere = mere.get_genome()
            # choix des points de croisement
            last_point = 0
            points = []
            if number_of_crossing_point == -1:
                # by default is max between the half of number of gene of 1 individual or 2.
                number_of_crossing_point = max(2, int(len(genome_mere)))
            for p in range(number_of_crossing_point):
                if last_point == len(pere.get_genome()) - 1:  # si on a deja atteint la fin
                    break
                last_point = np.random.randint(last_point + 1, len(pere.get_genome()))
                points.append(last_point)
            # on ajoute un dernier point pour boucler le truc
            points.append(len(mere.get_genome()))
            # croisement des individus
            last_point = 0  # debut du tableau
            sens = 1
            genome_fille = []
            genome_garcon = []
            for point in points:
                if sens == 1:
                    genome_fille = np.concatenate((genome_fille, genome_mere[last_point: point]))
                    genome_garcon = np.concatenate((genome_garcon, genome_pere[last_point: point]))
                    sens = -sens
                else:
                    genome_fille = np.concatenate((genome_fille, genome_pere[last_point: point]))
                    genome_garcon = np.concatenate((genome_garcon, genome_mere[last_point: point]))
                    sens = -sens
                last_point = point
            # ajout des individus à la nouvelle génération:
            self._add_new_individual_to_NEWpopulation(genome_garcon)
            self._add_new_individual_to_NEWpopulation(genome_fille)

    def create_next_generation(self):
        # start generation
        self.rank_population()  # order population
        # direct selection:
        self.elitism(_elitism)
        self.selection_by_rank(_by_rank)
        self.selection_by_fitness(_by_fitness)
        self.selection_by_tournament(_by_tournament)

        # mutations
        self.crossover_mutation(_mutation_crossover)
        self.inverse_mutation(_mutation_inverse)
        self.modify_mutation(_mutation_modify, _modify_range, _modify_number_type)

        # crossover
        self.crossover()

        # on echange les populations (pop+1)
        self.old_population = self.population
        self.population = []
        self.rank_population()  # order the output population

    # ---------- functions for direct selection ---------- #

    def elitism(self, number_individuals):
        for i in range(number_individuals):
            self.population = np.append(self.population, self.old_population[i])  # print(self._get_individual_info(self.old_population[i]))

    def selection_by_fitness(self, number_individuals):
        # creation du tableau de fitness
        fitness = np.array(([]))
        for ind in self.old_population:
            fitness = np.append(fitness, [ind.fitness])
        if np.min(fitness) < 0:
            fitness += abs(np.min(fitness))
        # proba
        proba = fitness / np.mean(fitness)
        proba = proba / sum(proba)
        # tire au sort X individus
        winners = np.random.choice(np.array(self.old_population), number_individuals, replace=False, p=proba)
        self.population = np.append(self.population, winners)

        # return np.random.choice(np.array(self.old_population), number_individuals, replace=False, p=proba)

    def selection_by_rank(self, number_individuals):
        # select one individual proportionally to the rank.
        # créé le tableau des probas, propotionnel au rang
        proba = []
        size = len(self.old_population)
        for i in range(size, 0, -1):
            proba.append(i)  # ajout du rang
        proba = proba / np.mean(proba)
        proba = proba / np.sum(proba)  # creation des probas (rang sur moyenne) ramener à 1
        # tire au sort x individu
        winners = np.random.choice(np.array(self.old_population), number_individuals, replace=False, p=proba)
        self.population = np.append(self.population, winners)

    def selection_by_tournament(self, number_individuals, tournament_participant=3):
        # take x random individual and append the winner to the next generation
        for i in range(number_individuals):
            individus_selected = []
            # choie des x individus
            individus_selected = np.random.choice(self.old_population, tournament_participant, replace=False)
            # tournois:
            winner = max(individus_selected, key=lambda x: x.fitness)
            # on ajoute le gagnant à la prochaine generation
            self.population = np.append(self.population, winner)  # return winner

    # ---------- function to visualize the evolution ---------- #

    def get_old_pop_infos(self):
        # make sure the population is already evaluate
        fitness = []
        for ind in self.old_population:
            fitness.append(ind.fitness)

        max_fitness = np.max(fitness)
        median_fitness = np.median(fitness)
        min_fitness = np.min(fitness)
        return max_fitness, median_fitness, min_fitness


def print_population(pop):
    i = 1
    print("old population: ")
    for ind in pop:
        if i > 10:
            print(".... and %d more" % (len(pop) - i))
            break
        print(" - rank " + str(i) + ": " + get_individual_infos(ind))
        i += 1


def get_individual_infos(individual):
    if individual.already_evaluate:
        fitness = str(individual.fitness)
    else:
        fitness = "(not evaluate)"
    genome = individual.get_genome()
    if len(genome) > 6:
        genome = str(genome[0:6]) + "..., len: " + str(len(genome)) + "."
    else:
        genome = str(genome) + ", len=" + str(len(genome)) + "."
    return " genome: " + genome + " Fitness: " + fitness


def multiple_evolutions(number_generation_by_iteration, number_of_iterations):
    global _file_setting
    load_file, _ = _file_setting

    ga = genetic_algorithm()

    all_max = []
    all_min = []
    all_median = []
    all_pop_len = []

    # process the genetics algorithms:
    print("starting evolution...")
    for i in range(number_of_iterations):
        all_max.append([])
        all_min.append([])
        all_median.append([])
        all_pop_len.append([])
        if load_file is None:
            ga.create_random_old_population(_individuals_count, debug=False)
        else:
            ga.load_old_population(load_file, debug=False)
        for number_gen in range(number_generation_by_iteration):
            ga.create_next_generation()
            max_val, median, min_val = ga.get_old_pop_infos()
            all_max[i].append(max_val)
            all_min[i].append(min_val)
            all_median[i].append(median)
            all_pop_len[i].append(len(ga.old_population))
        print("    finished evolution of " + str(i) + "/" + str(number_of_iterations))

    # for the graph now:
    x = np.linspace(0, number_generation_by_iteration, number_generation_by_iteration)
    max_fitness = []
    min_fitness = []
    median = []
    for gen in range(number_of_iterations):
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


def evolute(number_of_generation, auto_difference=0.1, auto_min_score=None):
    global _file_setting
    load_file, save_file = _file_setting

    genetic_algo = genetic_algorithm()

    if load_file is None:
        global _individuals_count
        genetic_algo.create_random_old_population(_individuals_count)
    else:
        genetic_algo.load_old_population(file_name=load_file)
    print_population(genetic_algo.old_population)

    print("\nstart population info: ")
    max_val, median, min_val = genetic_algo.get_old_pop_infos()
    print(" - best individual: " + get_individual_infos(genetic_algo.old_population[0]))
    print(" - fitness median: " + str(median))

    max_array = []
    min_array = []
    median_array = []
    len_pop_array = []

    g = s = p = 0
    # create x generation
    if number_of_generation >= 0:
        print("\nstarting evolution of %d generations" % number_of_generation)
        for i in tqdm(range(number_of_generation)):
            genetic_algo.create_next_generation()
            max_val, median, min_val = genetic_algo.get_old_pop_infos()
            max_array.append(max_val)
            median_array.append(median)
            min_array.append(min_val)
            len_pop_array.append(len(genetic_algo.old_population))
            g += 1
            if save_file is not None and g == s+5:
                genetic_algo.save_old_population(save_file, debug=False)
                s = g

    # auto
    if number_of_generation == -1:
        print("\nstarting auto evolution...")
        stop = False
        while not stop:
            genetic_algo.create_next_generation()
            max_val, median, min_val = genetic_algo.get_old_pop_infos()
            max_array.append(max_val)
            median_array.append(median)
            min_array.append(min_val)
            len_pop_array.append(len(genetic_algo.old_population))
            g += 1
            if save_file is not None and g == s+5:
                genetic_algo.save_old_population(save_file, debug=False)
                s = g
            # ecart entre le meilleur score et la median (sur 5 generations)
            e = np.mean(max_array[g-5:])-np.mean(median_array[g-5:])
            if auto_min_score is None:
                if e < auto_difference:
                    stop = True
                    number_of_generation = g
                elif s == p + 10:
                    print("    generation %d, difference: %g > %g" % (g, round(e, 3), auto_difference))
                    p = s
            elif max_val >= auto_min_score:
                if auto_min_score <= max_val:
                    if e < auto_difference:
                        stop = True
                        number_of_generation = g
                    elif s == p + 10:
                        print("    generation %d, difference: %g > %g" % (g, round(e, 3), auto_difference))
                        p = s
            elif s == p + 10:
                print("    generation %d, score: %g < %g" % (g, round(max_val, 3), round(auto_min_score, 3)))
                p = s

    print("auto evolution finished after %d generations" % number_of_generation)
    if save_file is not None:
        genetic_algo.save_old_population(save_file)

    print("\nfinal population info: ")
    max_val, median, min_val = genetic_algo.get_old_pop_infos()
    print(" - best individual: " + get_individual_infos(genetic_algo.old_population[0]))
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
