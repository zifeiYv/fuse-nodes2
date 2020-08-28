u"""
@Time   : 2020/6/18 3:02 \u4e0b\u5348
@Author : sunjiawei
@E-mail : j.w.sun1992@gmail.com
"""
import Levenshtein as Lvst, numpy as np, jieba


def lvst_dis(string1, string2):
    """Return the edit distance of two strings.
    
    Args:
        string1(str): 1st string
        string2(str): 2nd string
    
    Returns:
        Similarity value
    """
    dis = Lvst.distance(string1, string2)
    return 1 - dis / max(len(string1), len(string2))


def lcs(string1, string2):
    """Return the ratio of longest common string length to maximum length of original strings.
    
    Args:
        string1(str): 1st string
        string2(str): 2nd string
    
    Returns:
        Similarity value
    """
    if len(string1) == 0 or len(string2) == 0:
        return 0
    else:
        dp = [[0 for _ in range(len(string2) + 1)] for _ in range(len(string1) + 1)]
        max_len = 0
        for i in range(1, len(string1) + 1):
            for j in range(1, len(string2) + 1):
                if string1[i - 1] == string2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    max_len = max([max_len, dp[i][j]])
                else:
                    dp[i][j] = 0

        comm_sim = max_len / max(len(string1), len(string2))
        return comm_sim


def cosine_sim(string1, string2):
    """Return the cosine similarity of two strings.
    
    Args:
        string1(str): 1st string
        string2(str): 2nd string
    
    Returns:
        Similarity value
    """
    cut1 = jieba.cut(string1)
    cut2 = jieba.cut(string2)
    list_word1 = ','.join(cut1).split(',')
    list_word2 = ','.join(cut2).split(',')
    key_word = list(set(list_word1 + list_word2))
    word_vector1 = np.zeros(len(key_word))
    word_vector2 = np.zeros(len(key_word))
    for i in range(len(key_word)):
        for j in range(len(list_word1)):
            if key_word[i] == list_word1[j]:
                word_vector1[i] += 1

        for k in range(len(list_word2)):
            if key_word[i] == list_word2[k]:
                word_vector2[i] += 1

    cos_sim = float(np.dot(word_vector1, word_vector2) / (np.linalg.norm(word_vector1) * np.linalg.norm(word_vector2)))
    return cos_sim


def sims(string1, string2, methods=None):
    """Call three methods above to compute similarity.
    
    Args:
        string1(str): 1st string
        string2(str): 2nd string
        methods(dict): Weight dict, default None
    
    Returns:
        Similarity value
    """
    if string1 is None:
        return 0
    elif string2 is None:
        return 0
    else:
        sim = 0
        if not methods:
            methods = {'lvst':1, 
             'lcs':0,  'cos':0}
        else:
            if sum(methods.values()) != 1:
                raise UnNormalizedWeight("Sum of three methods' weight must be 1")
        for m in methods:
            if methods[m] != 0:
                if m == 'lvst':
                    sim += lvst_dis(string1, string2) * methods[m]
                elif m == 'lcs':
                    sim += lcs(string1, string2) * methods[m]
                else:
                    sim += cosine_sim(string1, string2) * methods[m]

        return sim


class UnNormalizedWeight(Exception):
    pass
