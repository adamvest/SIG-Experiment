'''
ML/DL SIG experiment 3
9/19
ChiMerge Value Reduction
'''


import numpy as np


class ChiMergeOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def parse_args(self):
        return self.parser.parse_args()


def test_intervals(data, intervals, start):
    k = len(set([x[1] for x in data]))
    chi_value = 0

    for i in range(2):
        splice1 = data[intervals[start+i]:intervals[start+i+1]]
        splice2 = data[intervals[start+i+1]:intervals[start+i+2]]

        for j in range(k):
            a_ij = len([x for x in splice1 if x[1] == j])
            print "a_ij", a_ij
            r_ij = intervals[start+i+1] - intervals[start+i]
            print "r_ij", r_ij
            c_j = len([x for x in splice1 if x[1] == j]) + len([x for x in splice2 if x[1] == j])
            print "c_j", c_j
            e_ij = max((r_ij * c_j) / (len(splice1) + len(splice2)), .1)
            print "e_ij", e_ij
            chi_value += (a_ij - e_ij) ** 2 / e_ij

    return chi_value


data = [(1.2, 1), (2.1, 1), (3.3, 0), (2.7, 0), (1.0, 1), (.8, 0), (2.2, 1), (4.3, 0), (.6, 0), (1.1, 1)]
data.sort(key=lambda x: x[0])
intervals = range(len(data) + 1)
finished = False

while finished == False:
    chi_values = []

    for i in range(len(data) - 1):
        print test_intervals(data, intervals, i)
        raw_input('stop')

    print chi_values
