import random
import numpy as np
from numpy.lib.function_base import place

# sum a list and stop to a specific index
def sum_stop(list, nb_end):
    val = 0.0
    for i, current_val in enumerate(list):
        val += current_val
        if i == nb_end:
            break
    return val

# object that represents a packet
class Packet():
    def __init__(self, proba, reward, indice, classe): 
        self.indice = indice         
        self.proba = proba 
        self.reward = reward
        self.classe = classe 

class Knapsack():
    def __init__(self, nb_packets, c, r, p, jobs):
        self.c = c
        self.r = r
        self.p = p
        self.jobs = jobs
        self.nb_packets = nb_packets
        # generate n packets according to probability distribution
        self.packets = self.gen_packets(nb_packets, self.p)
        # resolving problem with fractional knapsack
        self.fraction, self.alpha = self.get_fraction()

    # generate n packets 
    def gen_packets(self, nb, p):
        vec = []
        for i in range(nb):
            # generate a packet with probabilities p
            class_type = random.choices(self.jobs, k=1, weights=p)[0] - 1
            vec.append(Packet(p[class_type], self.r[class_type], i, self.jobs[class_type]))
        # sort vector depending on rewards
        vec.sort(key=lambda x: x.reward, reverse=True)
        return vec

    # gets the fraction of packet to add to saturate channel capacity
    def get_fraction(self):
        sum_prob = 0
        fraction = 0
        for i in range(len(self.jobs)):
            sum_prob += self.p[i]
            # if channel capacity is reaches 
            if sum_prob == self.c:
                fraction = 1
                break
            # if the capacity of channel is exceeded, we compute the fraction
            elif sum_prob > self.c:
                fraction = (self.c - sum_stop(self.p, i - 1)) / self.p[i]
                alpha = self.jobs[i] - 1
                break
        return fraction, alpha

    # compute best reward according to fraction
    def compute_bets_value(self):
        total_reward = 0
        for i, val in enumerate(self.packets):
            if val.classe > self.alpha:
                total_reward += self.fraction * val.reward
                break
            else:
                total_reward += val.reward
        return total_reward / len(self.packets)
     
    # estimate alpha : robins monro algo
    def search_alpha(self, stepsize, nb_iter, t, constant_step = 0):
        alpha_star = 0
        polyak = 0
        alpha = []

        i = 0
        for j in range(nb_iter):
            i += 1
            lambda_bar = self.expectation_lambda(t, alpha_star)

            # stepsizes chosen
            if(stepsize == "epsilon_step"):
                alpha_star = alpha_star + self.espsilon_stepsize(i) * (self.c - lambda_bar)
            elif(stepsize == "constant_step"):
                alpha_star = alpha_star + constant_step * (self.c - lambda_bar)
            elif(stepsize == "polyak_step"):
                if j == 0:
                    polyak = 1
                else:
                    polyak = sum(alpha) / t
                alpha_star = alpha_star + polyak * (self.c - lambda_bar)
                
            alpha.append(alpha_star)

            if(self.is_saturation_reaches(alpha_star, lambda_bar)):
                break

        return alpha_star, lambda_bar

    # decreasing stepsize
    def espsilon_stepsize(self, i):
        lambda_ = 0.9
        c = 6
        return (c / i**lambda_)

    # compute the saturation according to alpha star
    def is_saturation_reaches(self, alpha, val):
        #val  = sum_stop(self.p, int(alpha))
        saturation = val + (alpha - int(alpha)) * self.p[int(alpha) + 1]
        if(saturation >= self.c):
            return True
        return False

    # compute expectation of lamda
    def expectation_lambda(self, nb, alpha):
        tab = []
        for i in range(nb):
            tab.append(self.estimate_lambda(alpha, self.gen_packets(self.nb_packets, self.p)))
        return sum(tab) / len(tab)

    # get frequency admit packet / nb tot packet
    def estimate_lambda(self, alpha, packets):
        admit = 0
        reject = 0
        for i, val in enumerate(packets):
            if val.classe - 1 > int(alpha):
                reject += 1
            else:
                admit += 1
        return admit / (admit + reject)


if(__name__ == "__main__"):

    c = 0.34
    reward = [100,50,10,1] 
    prob = [1/3, 1/12, 1/4, 1/3]
    jobs = [1, 2, 3, 4]
    k = Knapsack(10000, c, reward, prob, jobs)

    # nb de fois génération de packet pour calculer expectation
    t = 50
    nb_iter = 50
    constant_stepsize = 1

    # exact value of alpha
    print(f"best value : {k.compute_bets_value()} fraction : {k.fraction}, alpha : {k.alpha}")

    # epsilon stepsize
    # plus lambda est grand plus on va avoir un dénominateur grand (notamment avec i qui augmente)
    # plus le stepsize sera petit
    # pour c plus il sera petit plus le stepsize diminuera
    # le temps de traitement est plus long pour arriver à saturation du réseau
    # Mais ce sera plus précis et l'approximation sera plus précise, demande plus d'itération
    print(k.search_alpha("epsilon_step", nb_iter, t))

    # le stepsize est toujours le même
    print(k.search_alpha("constant_step", nb_iter, t, constant_stepsize))

    print(k.search_alpha("polyak_step", 200, 10))