import os
import numpy as np
import sys
import warnings
root_path = os.getcwd()
sys.path.append(root_path)
from collections import Counter
import crystalbleu
from abc import ABC, abstractmethod
from nltk.translate.bleu_score import sentence_bleu,corpus_bleu
from rouge import Rouge
import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.CodeBLEU import bleu,weighted_ngram_match,syntax_match,dataflow_match
from utils.ast_utils import syntax_correct
from utils.human_eval.evaluation import evaluate_functional_correctness
from utils.apps.apps_evaluation import check_correctness, TIMEOUT
from utils.datasets_utils import DataItem

def calc_bleu_weights(l):
        if l == 1:
            bleu_weights = (1,)
        elif l == 2:
            bleu_weights = (0.5, 0.5)
        elif l == 3:
            bleu_weights = (0.33,0.33,0.34)
        else:
            bleu_weights = (0.25,0.25,0.25,0.25)
        return bleu_weights

class Oracle(ABC):

    def get_score(self, reference, hypothesis):
        pass

    def relative_degradation(self,y_o,y_m,y_gt):
        qt_o_gt = self.get_score(y_o, y_gt)
        qt_m_gt = self.get_score(y_m, y_gt)
        return (qt_o_gt - qt_m_gt)/(qt_o_gt + 1e-4)

    @abstractmethod
    def check_inconsistency(self, reference, hypothesis):
        pass

    def check_interesting(self, ref_logits, hypo_logits):
        pass

class StringOracle(Oracle):
    def __init__(self) -> None:
        self.last_prob = None
        super().__init__()

    def get_score(self, reference, hypothesis):
        return 1 if reference == hypothesis else 0 

    def check_inconsistency(self, reference, hypothesis):
        return True if reference != hypothesis else False

    def update_last_probs(self, probs):
        self.last_prob = probs

    def check_interesting(self, hypo_probs):
        if self.last_prob.shape != hypo_probs.shape:
            return True
        else:
            THRESHOLD = 0.1
            prob_delta = np.abs(np.array(self.last_prob) - np.array(hypo_probs))
            return True if np.max(prob_delta) > THRESHOLD else False

class BLEUOracle(Oracle):

    def __init__(self,n=4,threshold=0.9) -> None:
        self.n = n
        self.threshold = threshold

    def get_score(self,reference, hypothesis):
        min_len = min(len(reference.split()), len(hypothesis.split()))
        return corpus_bleu([[reference]],[hypothesis],weights=calc_bleu_weights(min_len))
    
    def check_inconsistency(self,reference, hypothesis):
        bleu_score = self.get_score(reference, hypothesis)
        return True if bleu_score < self.threshold else False


class ROUGEOracle(Oracle):

    def __init__(self,) -> None:
        pass

    def get_score(self,reference, hypothesis):
        rouge = Rouge()
        try:
            rouge_score = rouge.get_scores(hyps=[hypothesis], refs=[reference])
        except Exception:
            print(hypothesis)
            print(reference)
        return rouge_score[0]["rouge-l"]

    def check_inconsistency(self, reference, hypothesis):
        return super().check_inconsistency(reference, hypothesis)


class LevenshteinOracle(Oracle):

    def __init__(self, threshold=None) -> None:
        self.threshold = threshold

    def get_score(self,reference, hypothesis):
        return Levenshtein.ratio(reference, hypothesis)

    def check_inconsistency(self,reference, hypothesis):
        edit_sim = self.get_score(reference, hypothesis)
        return True if edit_sim < self.threshold else False


class CosineSimilarityOracle(Oracle):

    def __init__(self, threshold=None) -> None:
        self.threshold = threshold

    def get_score(self, reference, hypothesis):
        data = [reference, hypothesis]
        tfidf_vectorizer = TfidfVectorizer()
        vector_matrix = tfidf_vectorizer.fit_transform(data)
        cosine_similarity_matrix = cosine_similarity(vector_matrix)
        return cosine_similarity_matrix[0][1]

    def check_inconsistency(self,reference, hypothesis):
        cosine_sim = self.get_score(reference, hypothesis)
        return True if cosine_sim < self.threshold else False


class CodeBLEUOracle(Oracle):
    def __init__(self, lang='python',weights=(0.25,0.25,0.25,0.25),threshold=1.0) -> None:
        self.threshold = threshold
        self.lang = lang
        # the weights is used in code bleu. not for 4 grams
        self.weights = weights

    def get_score(self, ref, hypo):
        alpha,beta,gamma,theta = self.weights
        root_path = os.getcwd()
        hypothesis = [hypo]
        references = [[ref]]
        tokenized_hyps = [x.split() for x in hypothesis]
        tokenized_refs = [[x.split() for x in reference] for reference in references]

        min_len = min(len(hypo.split()), len(ref.split()))          
        ngram_match_score = bleu.corpus_bleu(tokenized_refs,tokenized_hyps,weights=calc_bleu_weights(min_len))
        # calculate weighted ngram match
        keywords_path = os.path.join(root_path,"utils/CodeBLEU",'keywords/'+self.lang+'.txt')
        keywords = [x.strip() for x in open(keywords_path, 'r', encoding='utf-8').readlines()]
        def make_weights(reference_tokens, key_word_list):
            return {token:1 if token in key_word_list else 0.2 \
                    for token in reference_tokens}
        tokenized_refs_with_weights = [[[reference_tokens, make_weights(reference_tokens, keywords)]\
                    for reference_tokens in reference] for reference in tokenized_refs]

        weighted_ngram_match_score = weighted_ngram_match.corpus_bleu(tokenized_refs_with_weights,tokenized_hyps)

        # calculate syntax match
        syntax_match_score = syntax_match.corpus_syntax_match(references, hypothesis, self.lang)

        # calculate dataflow match
        # We ignore the following warning 
        # `WARNING: There is no reference data-flows extracted from the whole corpus, and the data-flow match score degenerates to 0. Please consider ignoring this score.`

        dataflow_match_score = dataflow_match.corpus_dataflow_match(references, hypothesis, self.lang)

        # print('ngram match: {0}, weighted ngram match: {1}, syntax_match: {2}, dataflow_match: {3}'.\
        #                     format(ngram_match_score, weighted_ngram_match_score, syntax_match_score, dataflow_match_score))

        code_bleu_score = alpha*ngram_match_score\
                        + beta*weighted_ngram_match_score\
                        + gamma*syntax_match_score\
                        + theta*dataflow_match_score
        # print("xxx",ngram_match_score,weighted_ngram_match_score,syntax_match_score,dataflow_match_score,"xxx")
        # print('CodeBLEU score: ', code_bleu_score)
        return code_bleu_score

    def check_inconsistency(self,reference, hypothesis):
        code_bleu = self.get_score(reference, hypothesis)
        return True if code_bleu < self.threshold else False


class CrystalBLEU(Oracle):
    def __init__(self,dataset_name,k=500,threshold=None) -> None:
        self.dataset_name = dataset_name
        self.threshold = threshold
        training_set = self.get_training_set()
        frequencies = Counter(training_set) # tokenized_corpus is a list of strings
        self.trivially_shared_ngrams = dict(frequencies.most_common(k))

        # Calculate CrystalBLEU

    
    def get_training_set(self):
        if self.dataset_name == "humaneval":
            return np.load(os.path.join(root_path,f"data/humaneval_train_tokens.npy"))
        else:
            raise NotImplementedError(f"Current Crystal BLEU do not support {self.dataset_name}")


    def get_score(self, ref, hypo):
        hypothesis = [hypo]
        references = [[ref]]
        tokenized_hyps = [x for x in hypothesis]
        tokenized_refs = [[x for x in reference] for reference in references]
        min_len = min(len(hypo.split()), len(ref.split()))
        crystalBLEU_score = crystalbleu.corpus_bleu(tokenized_refs, tokenized_hyps, ignoring=self.trivially_shared_ngrams,
                                                    weights=calc_bleu_weights(min_len))
        return crystalBLEU_score

    def check_inconsistency(self,reference, hypothesis):
        crystal_codebleu = self.get_score(reference, hypothesis)
        return True if crystal_codebleu < self.threshold else False


class SyntaxOracle(Oracle):
    def check_inconsistency(self, hypothesis):
        if syntax_correct(hypothesis):
            return False
        else:
            return True

class Fact_Oracle(Oracle):
    def __init__(self) -> None:
        super().__init__()

    def check_inconsistency(self, applied_facts, attacked_facts): 
        applied_set = set([str(fact) for fact in applied_facts])
        attacked_set = set([str(fact) for fact in attacked_facts])
        missed_set = applied_set - attacked_set

        if len(missed_set) == 0:
            return False
        else:
            for missed_attack in missed_set:
                print(missed_attack)
            return True
        
    def check_attack_str_inconsistency(self, applied_set, attacked_facts): 
        attacked_set = set([str(fact) for fact in attacked_facts])
        missed_set = applied_set - attacked_set

        if len(missed_set) == 0:
            return False
        else:
            # for missed_attack in missed_set:
            #     # print(missed_attack)
            return True

class Pass_Oracle(Oracle):
    def __init__(self,ds_name) -> None:
        super().__init__()
        self.ds_name = ds_name
        self.last_pass_rate = None

    def update_last_pass_rate(self, rate):
        self.last_pass_rate = rate

    def get_pass_rate(self, **kw):
        """
        Use this function to get the pass rate of the attack cases.
        It's more flexible.
        """
        if self.ds_name == "humaneval":
            pass_rate = evaluate_functional_correctness([kw])
            return pass_rate
        elif self.ds_name == "APPS":
            prob_path = kw["prob_path"]
            debug = kw["debug"]
            generation=kw["generation"]
            curr_res = [-2]
            try:
                curr_res = check_correctness(prob_path,generation,TIMEOUT,debug)
                fixed = []
                for e in curr_res:
                    if isinstance(e, np.ndarray):
                       e = e.item(0)
                    if isinstance(e, np.bool_):
                        e = bool(e)
                    fixed.append(e)
                curr_res = fixed
                # if not np.all(curr_res):
                    # print(f"Results were not all True: {curr_res}")
            except Exception as excep:
                print(f"test framework exception = {repr(excep)}{excep}\n")
            finally:
                assert isinstance(curr_res, list)

            if len(curr_res) == 0:
                return -1.0
            curr_res = np.asarray(curr_res)
            pass_rate = np.mean(curr_res > 0)
        else:
            raise ValueError(f"Do not support such dataset:{self.ds_name}")
        return pass_rate

    
    def calc_orig_pass_rate(self, prompt, output, data_item:DataItem):
        """
        Use this function to get the pass rate of the original cases.
        """
        if self.ds_name == "humaneval":
            kw = dict(
                task_id=data_item.task_id,
                prompt=prompt,
                output=output,
                entry_point=data_item.entry_point, 
                test=data_item.tests
            )
            pass_rate = evaluate_functional_correctness([kw])
            return pass_rate
        elif self.ds_name == "APPS":
            prob_path = data_item.prob_path
            debug = False
            generation=output
            curr_res = [-2]
            try:
                curr_res = check_correctness(prob_path,generation,TIMEOUT,debug)
                fixed = []
                for e in curr_res:
                    if isinstance(e, np.ndarray):
                       e = e.item(0)
                    if isinstance(e, np.bool_):
                        e = bool(e)
                    fixed.append(e)
                curr_res = fixed
                # if not np.all(curr_res):
                    # print(f"Results were not all True: {curr_res}")
            except Exception as excep:
                print(f"test framework exception = {repr(excep)}{excep}\n")
            finally:
                assert isinstance(curr_res, list)

            if len(curr_res) == 0:
                return -1.0
            curr_res = np.asarray(curr_res)
            pass_rate = np.mean(curr_res > 0)
        else:
            raise ValueError(f"Do not support such dataset:{self.ds_name}")
        return pass_rate

    def check_inconsistency(self, **kw):
        pass_rate = self.get_pass_rate(**kw)
            # return acc
        if pass_rate == self.last_pass_rate:
            return False, pass_rate
        else:
            return True, pass_rate


oracles = {
    "edit_distance":LevenshteinOracle,
    "cosine_similarity":CosineSimilarityOracle,
    "bleu":BLEUOracle,
    "code_bleu":CodeBLEUOracle,
    "crystal_bleu":CrystalBLEU,
    "string": StringOracle,

}


if __name__ == "__main__":
    pass