"""
RBO: đo độ tương đồng giữa 2 lists. RBO trả về giá trị thuộc [0, 1]:
    1: 2 danh sách hoàn toàn giống nhau
    0: 2 danh sách hoàn toàn khác

Cách tính:
- Tạo tất cả cặp topic có thể (50C2). Với mỗi cặp, tính RBO 
"""




import numpy as np
import itertools
from collections import namedtuple

RBO = namedtuple('RBO', ['min', 'res', 'ext'])


def set_at_depth(lst, depth):
    return set(lst[:depth])


def agreement(list1, list2, depth):
    """Proportion of shared values between two sorted lists at given depth.

    >>> _round(agreement("abcde", "abdcf", 1))   => Depth = 1 thì đều giống nhau
    1.0
    >>> _round(agreement("abcde", "abdcf", 3))   => Depth = 3 thì abc khác abd
    0.667
    >>> _round(agreement("abcde", "abdcf", 4))   => Depth = 4 thì abcd và abdc, vẫn cùng 4 chữ đó
    1.0
    >>> _round(agreement("abcde", "abdcf", 5))
    0.8
    >>> _round(agreement([{1, 2}, 3], [1, {2, 3}], 1))
    0.667
    >>> _round(agreement([{1, 2}, 3], [1, {2, 3}], 2))
    1.0

    """
    set1, set2 = set_at_depth(list1, depth), set_at_depth(list2, depth)
    intersection = len(set1.intersection(set2))
    return 2 * intersection / (len(set1) + len(set2))


def rbo_ext(list1, list2, p):
    """
    NOTE: The doctests weren't verified against manual computations but seem
    plausible.
    >>> _round(rbo_ext("abcdefg", "abcdefg", .9))
    1.0
    >>> _round(rbo_ext("abcdefg", "bacdefg", .9))
    0.9
    """
    s, l = min(len(list1), len(list2)), max(len(list1), len(list2))
    x_l = len(set(list1[:l]).intersection(set(list2[:l])))
    x_s = len(set(list1[:s]).intersection(set(list2[:s])))
    
    sum1 = sum(p ** d * agreement(list1, list2, d) for d in range(1, l + 1))
    sum2 = sum(p ** d * x_s * (d - s) / s / d for d in range(s + 1, l + 1))
    term1 = (1 - p) / p * (sum1 + sum2)
    term2 = p ** l * ((x_l - x_s) / l + x_s / s)
    return term1 + term2

def rbo(list1, list2, p):
    """Complete RBO analysis (lower bound, residual, point estimate).
    >>> lst1 = [{"c", "a"}, "b", "d"]
    >>> lst2 = ["a", {"c", "b"}, "d"]
    >>> _round(rbo(lst1, lst2, p=.9))
    RBO(min=0.489, res=0.477, ext=0.967)
    """
    if not 0 <= p <= 1:
        raise ValueError("p must be between 0 and 1")
    return rbo_ext(list1, list2, p)



class InvertedRBO:
    def __init__(self, topics):
        """
        :param topics: a list of lists of words
        """
        self.topics = topics

    def score(self, topk=10, weight=0.9):
        """
        :param weight: p (float), default 1.0: Weight of each agreement at
         depth d: p**(d-1). When set to 1.0, there is no weight, the rbo
         returns to average overlap.
        :return: rank_biased_overlap over the topics
        """
        if topk > len(self.topics[0]):
            raise ValueError('topk exceeds topic word count')
        
        rbo_scores = []
        for list1, list2 in itertools.combinations(self.topics, 2):
            rbo_val = rbo(list1[:topk], list2[:topk], weight)
            rbo_scores.append(rbo_val)
        
        return 1 - np.mean(rbo_scores)