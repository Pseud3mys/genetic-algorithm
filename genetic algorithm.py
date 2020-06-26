"""
a genetic algorithm who work with a neural network to solve a problem
"""
import numpy as np
import os

# default parameters for the individuals:
default_params_individuals = [30, 20, None, None]

# default parameters for probability:
default_params_probability = [[0, True], 0.1, 0.05, 0.05, 0.08, 0.05]


class genetic_algorithm:
    def __init__(self, params_for_individuals, params_for_probability):
        self._save_dir = os.getcwd() + "\\saves\\"

        # individual params:
        self._start_individuals_count = params_for_individuals[0]
        self._mean_individuals_count = params_for_individuals[1]
        self._individuals_class = params_for_individuals[2]
        self._individuals_class_param = params_for_individuals[3]

        # generation params:
        self._default_generation_params = params_for_probability
        # probability selection:
        self._probability_elitism = params_for_probability[0][0]
        self._force_elitism = params_for_probability[0][1]
        self._probability_selection_by_rank = params_for_probability[1]
        self._probability_selection_by_fitness = params_for_probability[2]
        self._probability_selection_by_tournament = params_for_probability[3]
        # probability mutation
        self._probability_mutation = params_for_probability[4]
        self._probability_mutation_multi_points = params_for_probability[5]

        # print some infos/debug:
        print("\n-------- genetic algorithm infos --------")
        # create a fake individual for the debug
        alfred = self._individuals_class(self._individuals_class_param, None)
        print("\n - individuals:")
        print("          - number the first generation: " + str(self._start_individuals_count))
        print("          - number by generation: ~" + str(self._mean_individuals_count))
        print("          - number of genes: " + str(len(alfred.get_genome())))
        print("          - class: " + str(self._individuals_class))
        print("\n - generations:")
        print("          - probability elitism: " + str(self._probability_elitism))
        print("          - probability selection by rank: " + str(self._probability_selection_by_rank))
        print("          - probability selection by fitness: " + str(self._probability_selection_by_fitness))
        print("          - probability selection tournament: " + str(self._probability_selection_by_tournament))
        print("          - probability mutation: " + str(self._probability_mutation))
        print("          - probability mutation multi points: " + str(self._probability_mutation_multi_points))
        print("          - check probability: " + self._test_probability())
        print("------------------------------------------\n\n")
        # create the pop
        self.old_population = np.array(([]))
        self.population = np.array(([]))

    def _add_new_individual_to_population(self, genome):
        new_individual = self._individuals_class(self._individuals_class_param, genome)
        self.population = np.append(self.population, new_individual)

    def _test_probability(self):
        proba_sum = self._probability_mutation + self._probability_selection_by_tournament + self._probability_selection_by_rank
        proba_sum += self._probability_elitism + self._probability_mutation_multi_points + self._probability_selection_by_fitness
        if proba_sum > 1:
            raise ValueError(" the sum of the probability is greater than 1")
        elif proba_sum > 0.8:
            return "to modify (" + str(proba_sum) + " > 0.8) !"
        elif proba_sum > 0.5:
            return "okay (" + str(proba_sum) + " > 0.5)"
        else:
            return "good (" + str(proba_sum) + " < 0.5)"

    def _gestation_proba(self, probability):
        number_individuals = 0
        for i in range(len(self.old_population)):
            r = np.random.choice(np.random.random(5))
            if r < probability:
                number_individuals += 1
        return number_individuals

    # ---------- functions for the save and first population ---------- #

    def create_random_old_population(self, debug=True):
        # on reset les pop:
        self.population = np.array(([]))
        self.old_population = np.array(([]))
        for i in range(self._start_individuals_count):
            # create new people with a random genome
            new_individu = self._individuals_class(self._individuals_class_param, None)
            self.old_population = np.append(self.old_population, new_individu)
        # finally we rank the pop randomly generated
        self.rank_population()
        if debug:
            print("SAVED " + str(len(self.old_population)) + " individus with " + str(len(self.old_population[0].get_genome())) +
                  "gene each.")

    def load_old_population(self, file_name, debug=True):
        # on reset les pop:
        self.population = np.array(([]))
        self.old_population = np.array(([]))
        genomes_array = np.load(self._save_dir + file_name)
        for genome in genomes_array:
            # create new people with a predefine genome
            new_individu = self._individuals_class(self._individuals_class_param, genome)
            self.old_population = np.append(self.old_population, new_individu)
        # finally we rank the pop loaded
        self.rank_population()
        if debug:
            print("LOADED " + str(len(genomes_array)) + " individuals with " + str(len(genomes_array[0])) + " gene each.")

    def save_old_population(self, file_name, debug=True):
        genomes_array = []
        for ind in self.old_population:
            genomes_array.append(ind.get_genome())
        np.save(self._save_dir + file_name, genomes_array)
        if debug:
            print("SAVED " + str(len(genomes_array)) + " individus with " + str(len(genomes_array[0])) + " gene each.")

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
        for ind in self.old_population:
            if not ind.already_evaluate:
                i += 1
                ind.evaluate()
                ind.already_evaluate = True
        # les trie en fonction de leur fitness
        self.old_population = sorted(self.old_population, key=lambda x: x.fitness, reverse=True)

    def mutation(self, probability):
        number_individuals = self._gestation_proba(probability)
        for i in range(number_individuals):
            # crossover between an old mother and a new random father
            mere = np.random.choice(self.old_population)  # an individual chose randomly in the old population
            pere = self._individuals_class(self._individuals_class_param, None)  # new random individual
            # we take the genome of the parents:
            genome_mere = mere.get_genome()
            genome_pere = pere.get_genome()
            # taille du point: un nombre random entre 1 et 10% du nombre de gene du genome
            taille_du_point = max(1, int(np.random.randint(0, int(len(genome_mere) * 0.1 + 1))))
            # puis la position du point
            position_du_point = np.random.randint(1, len(genome_mere) - taille_du_point)
            # on fait le genome fille en remplacant tous les genes sur la taille du point par les genes de l'individu random
            genome_fille = np.concatenate((genome_mere[:position_du_point],
                                           genome_pere[position_du_point: position_du_point + taille_du_point],
                                           genome_mere[position_du_point + taille_du_point:]))
            self._add_new_individual_to_population(genome_fille)

    def mutation_multi_points(self, probability, number_of_crossing_point=-1):
        number_individuals = self._gestation_proba(probability)
        for i in range(number_individuals):
            # number of crossing point
            # crossover between an old mother and a new random father
            mere = np.random.choice(self.old_population)
            pere = self._individuals_class(self._individuals_class_param, None)
            genome_mere = mere.get_genome()
            genome_pere = pere.get_genome()
            # choix des points de croisement
            last_point = 0
            points = []
            if number_of_crossing_point == -1:
                # by default is max between the half of number of gene of 1 individual or 2.
                number_of_crossing_point = max(2, int(len(genome_mere) / 2))
            for p in range(number_of_crossing_point):
                if last_point == len(pere.get_genome()) - 1:  # on a deja atteint la fin
                    break
                last_point = np.random.randint(last_point + 1, len(pere.get_genome()))
                points.append(last_point)
            # on ajoute un dernier point pour boucler le truc
            points.append(len(mere.get_genome()))
            # croisement des individus
            last_point = 0  # debut du tableau
            sens = 1
            genome_fille = []
            for point in points:
                if sens == 1:
                    genome_fille = np.concatenate((genome_fille, genome_mere[last_point: point]))
                    sens = -sens
                else:
                    genome_fille = np.concatenate((genome_fille, genome_pere[last_point: point]))
                    sens = -sens
                last_point = point
            # ajout du mutant à la nouvelle génération:
            self._add_new_individual_to_population(genome_fille)

    def crossover(self, number_of_crossing_point=-1):
        while len(self.population) < self._mean_individuals_count:
            # croise les individus deja selectionné pour le génération suivant entre eux
            pere = np.random.choice(self.old_population)
            mere = np.random.choice(self.old_population)
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
            self._add_new_individual_to_population(genome_garcon)
            self._add_new_individual_to_population(genome_fille)

    def create_next_generation(self, special_generation_params=None):
        if special_generation_params is None:
            generation_params = self._default_generation_params
        else:
            generation_params = special_generation_params
        # read parameters
        # probability
        self._probability_elitism = generation_params[0][0]
        self._force_elitism = generation_params[0][1]
        self._probability_selection_by_rank = generation_params[1]
        self._probability_selection_by_fitness = generation_params[2]
        self._probability_selection_by_tournament = generation_params[3]
        # probability mutation
        self._probability_mutation = generation_params[4]
        self._probability_mutation_multi_points = generation_params[5]

        # start generation
        self.rank_population()  # order population
        # direct selection:
        self.elitism(self._probability_elitism, self._force_elitism)
        self.selection_by_rank(self._probability_selection_by_rank)
        self.selection_by_fitness(self._probability_selection_by_fitness)
        self.selection_by_tournament(self._probability_selection_by_tournament)
        # crossover
        self.crossover()
        # mutation
        self.mutation(self._probability_mutation)
        self.mutation_multi_points(self._probability_mutation_multi_points)

        # on echange les populations
        self.old_population = self.population
        self.population = []
        self.rank_population()  # order the output population

    # ---------- functions for direct selection ---------- #

    def elitism(self, probability, force=True):
        number_individuals = self._gestation_proba(probability)
        if force and number_individuals < 1:
            number_individuals = 1
        for i in range(number_individuals):
            self.population = np.append(self.population, self.old_population[i])  # print(self._get_individual_info(self.old_population[i]))

    def selection_by_fitness(self, probability):
        number_individuals = self._gestation_proba(probability)
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

    def selection_by_rank(self, probability):
        number_individuals = self._gestation_proba(probability)
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

    def selection_by_tournament(self, probability, tournament_participant=3):
        number_individuals = self._gestation_proba(probability)
        # take x random individual and append the winner to the next generation
        for i in range(number_individuals):
            individus_selected = []
            selected_count = 0
            # choie des x individus
            while selected_count < tournament_participant:
                r = np.random.randint(0, len(self.old_population) - 1)
                if individus_selected.count(self.old_population[r]) == 0:
                    individus_selected.append(self.old_population[r])
                    selected_count += 1
            # tournois:
            finalist1 = max(individus_selected[:int(tournament_participant / 2)], key=lambda x: x.fitness)
            finalist2 = max(individus_selected[int(tournament_participant / 2):tournament_participant], key=lambda x: x.fitness)
            winner = max([finalist1, finalist2], key=lambda x: x.fitness)
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
