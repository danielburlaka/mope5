import random
import math
import numpy as np

from _pydecimal import Decimal
from scipy.stats import f, t
from functools import reduce
from itertools import compress


x1min, x2min, x3min = -10, -2, -1
x1max, x2max, x3max = 10, 2, 4

x_average_min = (x1min + x2min + x3min) / 3
x_average_max = (x1max + x2max + x3max) / 3

x0_i = [(x1max+x1min)/2, (x2max+x2min)/2, (x3max+x3min)/2]
det_x_i = [(x1min - x0_i[0]), (x2min-x0_i[1]), (x3min-x0_i[2])]

l = 1.215
m = 3
N = 15

ymin = round(200 + x_average_min)
ymax = round(200 + x_average_max)


def getStudentValue(f3, q):
    return Decimal(abs(t.ppf(q/2,f3))).quantize(Decimal('.0001'))

def getFisherValue(f3,f4, q):
    return Decimal(abs(f.isf(q,f4,f3))).quantize(Decimal('.0001'))

def criteriaStudent(m, N, y_table, beta_coefficients):
    print("\nValidation of regression coefficients using Student's criteria: ".format(m, N))
    average_variation = np.average(list(map(np.var, y_table)))

    variation_beta_s = average_variation/N/m
    standard_deviation_beta_s = math.sqrt(variation_beta_s)
    t_i = np.array([abs(beta_coefficients[i])/standard_deviation_beta_s for i in range(len(beta_coefficients))])
    f3 = (m-1)*N
    q = 0.05

    t = getStudentValue(f3, q)
    importance = [True if el > t else False for el in list(t_i)]

    print("Coefficient estimates βs: " + ", ".join(list(map(lambda x: str(round(float(x), 3)), beta_coefficients))))
    print("Coefficient ts:         " + ", ".join(list(map(lambda i: "{:.2f}".format(i), t_i))))
    print("f3 = {}; q = {}; table = {}".format(f3, q, t))
    beta_i = ["β0", "β1", "β2", "β3", "β12", "β13", "β23", "β123", "β11", "β22", "β33"]
    importance_to_print = ["important" if i else "unimportant" for i in importance]
    to_print = map(lambda x: x[0] + " " + x[1], zip(beta_i, importance_to_print))
    x_i_names = list(compress(["", "x1", "x2", "x3", "x12", "x13", "x23", "x123", "x1^2", "x2^2", "x3^2"], importance))
    betas_to_print = list(compress(beta_coefficients, importance))
    print(*to_print, sep="; ")
    equation = " ".join(["".join(i) for i in zip(list(map(lambda x: "{:+.2f}".format(x), betas_to_print)),x_i_names)])
    print("The regression equation without insignificant members: y = " + equation)
    return importance

def createFactorTable(raw_array):
    return [row + [row[0] * row[1], row[0] * row[2], row[1] * row[2], row[0] * row[1] * row[2]]
     + list(map(lambda x: round(x ** 2, 5), row))
     for row in raw_array]

def m_ij(*arrays):
    return np.average(reduce(lambda accum, el: accum*el, arrays))

def x_i(i):
    try:
        assert i <= 10
    except:
        raise AssertionError("it should be smaller or equal 10")
    with_null_factor = list(map(lambda x: [1] + x, createFactorTable(raw_factors_table)))
    res = [row[i] for row in with_null_factor]
    return np.array(res)

def cochranValue(f1, f2, q):
    partResult1 = q / f2 # (f2 - 1)
    params = [partResult1, f1, (f2 - 1) * f1]
    fisher = f.isf(*params)
    result = fisher/(fisher + (f2 - 1))
    return Decimal(result).quantize(Decimal('.0001'))

def cochranCriteria(m, N, y_table):
    print("\nChecking the uniformity of dispersions by Cochran criteria: ".format(m, N))
    y_variations = [np.var(i) for i in y_table]
    max_y_variation = max(y_variations)
    gp = max_y_variation/sum(y_variations)
    f1 = m - 1
    f2 = N
    p = 0.95
    q = 1-p
    gt = cochranValue(f1,f2, q)
    print("Gp = {}; Gt = {}; f1 = {}; f2 = {}; q = {:.2f}".format(gp, gt, f1, f2, q))
    if gp < gt:
        print("Gp < Gt => dispersions are uniform ")
        return True
    else:
        print("Gp > Gt => dispersions are uneven ")
        return False

def calculateTheoretical_Y(x_table, b_coefficients, importance):
    x_table = [list(compress(row, importance)) for row in x_table]
    b_coefficients = list(compress(b_coefficients, importance))
    y_vals = np.array([sum(map(lambda x, b: x*b, row, b_coefficients)) for row in x_table])
    return y_vals

def fisherCriteria(m, N, d, naturalized_x_table, y_table, b_coefficients, importance):
    f3 = (m - 1) * N
    f4 = N - d
    q = 0.05

    theoretical_y = calculateTheoretical_Y(naturalized_x_table, b_coefficients, importance)
    theoretical_values_to_print = list(zip(map(lambda x: "x1 = {0[1]}, x2 = {0[2]}, x3 = {0[3]}".format(x),
    naturalized_x_table), theoretical_y))

    y_averages = np.array(list(map(np.average, y_table)))
    s_ad = m/(N-d)*(sum((theoretical_y-y_averages)**2))
    y_variations = np.array(list(map(np.var, y_table)))
    s_v = np.average(y_variations)
    f_p = float(s_ad/s_v)
    f_t = getFisherValue(f3, f4, q)

    print("\nFisher's model adequacy check:".format(m, N))
    print("Theoretical values of Y for different combinations of factors:")
    print("\n".join(["{arr[0]}: y = {arr[1]}".format(arr=el) for el in theoretical_values_to_print]))
    print("Fp = {}, Ft = {}".format(f_p, f_t))
    print("Fp < Ft => the model is adequate" if f_p < f_t else "Fp > Ft => the model is inadequate")
    return True if f_p < f_t else False

rawNaturalizedFactorsTable = [[x1min, x2min, x3min],
                              [x1min, x2max, x3max],
                              [x1max, x2min, x3max],
                              [x1max, x2max, x3min],

                              [x1min, x2min, x3max],
                              [x1min, x2max, x3min],
                              [x1max, x2min, x3min],
                              [x1max, x2max, x3max],

                              [-l*det_x_i[0]+x0_i[0], x0_i[1], x0_i[2]],
                              [l*det_x_i[0]+x0_i[0], x0_i[1], x0_i[2]],
                              [x0_i[0], -l*det_x_i[1]+x0_i[1], x0_i[2]],
                              [x0_i[0], l*det_x_i[1]+x0_i[1], x0_i[2]],
                              [x0_i[0], x0_i[1], -l*det_x_i[2]+x0_i[2]],
                              [x0_i[0], x0_i[1], l*det_x_i[2]+x0_i[2]],

                              [x0_i[0],      x0_i[1],     x0_i[2]]]

raw_factors_table = [[-1, -1, -1],
                     [-1, +1, +1],
                     [+1, -1, +1],
                     [+1, +1, -1],

                     [-1, -1, +1],
                     [-1, +1, -1],
                     [+1, -1, -1],
                     [+1, +1, +1],

                     [-1.215, 0, 0],
                     [+1.215, 0, 0],
                     [0, -1.215, 0],
                     [0, +1.215, 0],
                     [0, 0, -1.215],
                     [0, 0, +1.215],

                     [0, 0, 0]]

factorsTable = createFactorTable(raw_factors_table)

for row in factorsTable:
    print(row)
naturalizedFactorsTable = createFactorTable(rawNaturalizedFactorsTable)
withNullFactor = list(map(lambda x: [1] + x, naturalizedFactorsTable))

y_arr = [[random.randint(ymin, ymax) for _ in range(m)] for _ in range(N)]

while not cochranCriteria(m, N, y_arr):
    m += 1
    y_arr = [[random.randint(ymin, ymax) for _ in range(m)] for _ in range(N)]

y_i = np.array([np.average(row) for row in y_arr])

coefficients = [[m_ij(x_i(column)*x_i(row)) for column in range(11)] for row in range(11)]

freeValues = [m_ij(y_i, x_i(i)) for i in range(11)]

betaCoefficients = np.linalg.solve(coefficients, freeValues)
print(list(map(int,betaCoefficients)))

importance = criteriaStudent(m, N, y_arr, betaCoefficients)
d = len(list(filter(None, importance)))
fisherCriteria(m, N, d, naturalizedFactorsTable, y_arr, betaCoefficients, importance)