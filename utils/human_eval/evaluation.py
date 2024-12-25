import os
import sys
sys.path.append(os.getcwd())
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union, Iterable, Dict
import itertools
import numpy as np
from tqdm import tqdm
from utils.human_eval.data import write_jsonl
from utils.human_eval.execution import check_correctness


def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int 
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


def evaluate_functional_correctness(
    solutions: List,
    k: List[int] = [1],
    n_workers: int = 4,
    timeout: float = 3.0,
):

    # Check the generated samples against test suites.
    with ThreadPoolExecutor(max_workers=n_workers) as executor:

        futures = []
        results = []

        print("Reading samples...")
        assert len(solutions) == 1 ,f"In our study , we only generate 1 code sample instead of {len(solutions)}"
        solution_item = solutions[0]
        tests = solution_item["test"]
        for t_id, test in enumerate(tests):
            # solution_item is a dict like {"prompt":xx, "test":xx, "entry_point":xx, "output":xx}
            task_id = solution_item["task_id"]
            args = (task_id,solution_item["prompt"],solution_item["output"],solution_item["entry_point"],test, timeout,t_id)
            future = executor.submit(check_correctness, *args)
            futures.append(future)

        print("Running test suites...")
        for future in as_completed(futures):
            result = future.result()
            # return result is a dict like {"task_id":xxx, passed:True, result:"passed",test_id=xx}
            results.append(result["passed"])

    total = len(results)
    correct = sum(results)
    if total == 0:
        return -1
    else:
        return correct / total
