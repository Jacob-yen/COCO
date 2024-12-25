import numpy as np
import uuid
import random
from typing import List,Dict
import os
import sys
sys.path.append(os.getcwd())
from utils.logging_utils import MyLogger
import logging
from collections import defaultdict
# from gramformer import Gramformer
# import spacy
# spacy.load('en_core_web_sm')
# gf = Gramformer(models = 1, use_gpu=False) # 1=corrector, 2=detector

# class Corrector:
#     @staticmethod
#     def correct_sentences(s):
#         sentncs = gf.correct(s, max_candidates=1)
#         return list(sentncs)[0]

class Mutant:
    def __init__(self, id=None,entry_point=None,tests=None) -> None:
        # fact nodes which had been used
        self.applied_facts = []
        self.applied_facts_dict = defaultdict(list)
        self.id = id
        self.entry_point = entry_point
        self.tests = tests
        self.selected = 0
        self.uid = None
    
    def update_usage(self):
        self.selected += 1

    def update_applied_facts(self,mutator_name, fact):
        self.applied_facts.append(fact)
        self.applied_facts_dict[mutator_name].append(fact)

    def fork_mutant(self,mutant):
        for mu_name, facts in mutant.applied_facts_dict.items():
            for fact in facts:
                self.applied_facts.append(fact)
                self.applied_facts_dict[mu_name].append(fact)

    
    @property
    def score(self):
        """the score measuring that whether mutant is worth being selected next time. 

        Returns:
            float: score
        """
        return 1.0 / (self.selected + 1)

    @property
    def order(self):
        return len(self.applied_facts)

    def __str__(self) -> str:
        return f"mutant: {self.id}. score: {self.score}"


class SeedPool:
    def __init__(self,capacity=101):
        """Class definition of basic seed pool.

        Args:
            seeds_names (List[str], optional): The file name of the seeds. Defaults to None.
            capacity (int, optional): the capacity of the seed pool. Defaults to 100.
        """
        self.capacity = capacity
        self._mutants = []

    @property
    def mutants(self):
        return self._mutants

    def sort_mutants(self):
        self._mutants.sort(key=lambda mutant: mutant.score, reverse=True)

    @property
    def pool_size(self):
        return len(self._mutants)

    def add_mutant(self, mutant):
        if self.pool_size >= self.capacity:
            self.pop_one_mutant(-1)
        mutant.uid = self.pool_size
        self._mutants.append(mutant)
    
    def pop_one_mutant(self,k):
        """Randomly pop one mutant from the seed pool
        """
        self._mutants.pop(k)
        
    def is_full(self):
        """To check whether the seed pool is full
        """
        if len(self._mutants) >= self.capacity:
            return True
        else:
            return False

    def choose_mutant(self):
        pass

class RouletteSeedPool(SeedPool):
    """Roulette strategy based Seed Pool
    """
    def choose_mutant(self)->Mutant:
        sum = 0
        for mutant in self._mutants:
            sum += mutant.score
        rand_num = np.random.rand() * sum
        for mutant in self._mutants:
            if rand_num <= mutant.score:
                return mutant
            else:
                rand_num -= mutant.score

class RandomSeedPool(SeedPool):
    """Random strategy based Seed Pool
    """
    def choose_mutant(self)->Mutant:
        r = np.random.choice(a = list(range(self.pool_size)),replace=False)
        return self._mutants[r]
    
class Mutator:
    def __init__(self, name):
        self.name = name
        self.total = 0
        self.success = 0
        self.uid = None

    @property
    def score(self, epsilon=1e-7):
        return self.success / (self.total + epsilon)

    def add_success(self):
        self.success += 1
    
    def add_total(self):
        self.total += 1

    def __str__(self) -> str:
        return f"mutator: {self.name}. Score: {self.score}"

class Selection:
    """Base Selection Class
    """
    def __init__(self,mutate_ops=None) -> None:
        # shuffle the mutate ops when initialization
        random.shuffle(mutate_ops)
        self._mutators = [Mutator(name=op) for op in mutate_ops]

    @property
    def mutator_count(self):
        return len(self._mutators)

    @property
    def mutators(self):
        return self._mutators

    def choose_mutator(self,):
        pass

class RandomSelection(Selection):
    def __init__(self, mutate_ops=None) -> None:
        logger = logging.getLogger('mylogger')
        logger.info("Selelcting mutator via random strategy")
        super().__init__(mutate_ops)

    def choose_mutator(self,**kwargs):
        return self._mutators[np.random.randint(0, len(self._mutators))]

class BanditsSelection(Selection):
    """
    The Epsilon-Greedy policy will choose a random action with probability
    epsilon and take the best apparent approach with probability 1-epsilon. If
    multiple actions are tied for best choice, then a random action from that
    subset is selected.
    """
    def __init__(self, mutate_ops=None,epsilon=0.3) -> None:
        logger = logging.getLogger('mylogger')
        logger.info("Selelcting mutator via bandit")
        self.epsilon = epsilon
        super().__init__(mutate_ops)


    def choose_mutator(self,**kwargs):
        if np.random.random() < self.epsilon:
            return self._mutators[np.random.randint(0, len(self._mutators))]
        else:
            scores = [m.score for m in self._mutators]
            max_value = np.max(scores)
            candidates = []
            for idx in range(len(self.mutators)):
                if self._mutators[idx].score == max_value:
                    candidates.append(idx)
            if len(candidates) == 1:
                return self._mutators[candidates[0]]
            else:
                return self._mutators[np.random.choice(candidates)]

class MCMCSelection(Selection):

    def __init__(self, mutate_ops=None,epsilon=0.3) -> None:
        self.epsilon = epsilon
        logger = logging.getLogger('mylogger')
        logger.info("Selelcting mutator via mcmc")
        super().__init__(mutate_ops)
        self.p = 1 / len(mutate_ops)


    def sort_mutators(self):
        random.shuffle(self._mutators)
        self._mutators.sort(key=lambda mutator: mutator.score, reverse=True)

    def index(self, mutator_name):
        for i, mu in enumerate(self._mutators):
            if mu.name == mutator_name:
                return i
        else:
            logger = logging.getLogger('mylogger')
            logger.info(f"Can't found the mutator {mutator_name} in selection list.")
            raise ValueError(f"Can't found the mutator {mutator_name} in selection list.")

    def choose_mutator(self,**kwargs):
        mu1 = kwargs['last_used_mutator']
        if mu1 is None:
            # which means it's the first mutation
            return self._mutators[np.random.randint(0, len(self._mutators))]
        else:
            self.sort_mutators()
            k1 = self.index(mu1.name)
            k2 = -1
            prob = 0
            while np.random.rand() >= prob:
                k2 = np.random.randint(0, len(self._mutators))
                prob = (1 - self.p) ** (k2 - k1)
            mu2 = self._mutators[k2]
            return mu2

Strategy = {
    "mcmc":MCMCSelection,
    "bandit":BanditsSelection,
    "random":RandomSelection
}


if __name__ == "__main__":
    mutators_ops = ["m1","m2","m3","m4"]
    # Test RandomSelection
    test_size = 100
    # print("=====Test Random Selection=====")
    # random_selection = RandomSelection(mutators_ops)
    # for i in range(test_size):
    #     mu = random_selection.choose_mutator()
    #     print(i,str(mu))

    # print("=====Test Bandit Selection=====")
    # bandit_selection = BanditsSelection(mutators_ops)
    # for i in range(test_size):
    #     mu = bandit_selection.choose_mutator()
    #     mu.add_total()
    #     if np.random.rand() >= 0.5:
    #         mu.add_success()
    #     print(i,str(mu))

    # print("=====Test MCMC Selection=====")
    # mcmc_selection = MCMCSelection(mutators_ops)
    # last_mu = None
    # for i in range(test_size):
    #     mu = mcmc_selection.choose_mutator(last_mu)
    #     last_mu = mu.name
    #     mu.add_total()
    #     if np.random.rand() >= 0.5:
    #         mu.add_success()
    #     print(i,str(mu))

    seedpool = RouletteSeedPool()
    cnt = 1
    seedpool.add_mutant(Mutant(id=cnt))
    cnt += 1

    for _ in range(50):
        mutant = seedpool.choose_mutant()
        mutant.update_usage()
        print(str(mutant))
        if np.random.rand() >= 0.5:
            seedpool.add_mutant(Mutant(id=cnt))
            cnt += 1

