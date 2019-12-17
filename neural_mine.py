import random
from math import exp, sqrt
import numpy
import data_arr


def forward(l1, l2, w):
    '''заменить на произведение матриц'''
    ''' размер входящего слоя'''
    insize = len(w)
    '''научиться находить размер внутреннего массива '''
    outsize = len(w[0])
    outi = 0
    while outi < outsize:
        ini = 0
        l2[outi] = 0
        while ini < insize:
            l2[outi] = l2[outi] + l1[ini] * w[ini][outi]
            ini += 1
        l2[outi] = 1 / (1 + exp(-1 * l2[outi]))
        outi += 1
    return l2


def findErrorOut(res, idl):
    print("Result: {}" .format(res))
    print("IDL: {}" .format(idl))
    out = []
    for x in range(0, len(res)):
        #er = (idl[x] - res[x]) * res[x] * (1 - res[x])
        er = pow((res[x] - idl[x]), 2)
        out.append(er)
    print("Err: {}" .format(out))
    return out


def findError(li, w, lo):
    print("Out neur err: {}".format(lo))
    print("In neur weight: {}".format(w))
    '''lo - ошибки выходных нейронов '''
    insize = len(w)
    '''научиться находить размер внутреннего массива '''
    outsize = len(w[0])
    ini = 0
    while ini < insize:
        outi = 0
        li[ini] = 0
        while outi < outsize:
            li[ini] = li[ini] + w[ini][outi] * lo[outi]
            outi += 1
        ini += 1
    return li


def backward(l1, l2, le, w, k):
    insize = len(w)
    outsize = len(w[0])
    outi = 0
    while outi < outsize:
        ini = 0
        while ini < insize:
            w[ini][outi] = w[ini][outi] + k * le[outi] * l1[ini] * l2[outi] * (1 - l2[outi])
            ini += 1
        outi += 1
    return w


""" входной слой """

# первый элемент верт второй горизонт


"""скрытый слой 1"""
"""перейти за заполнение массива сразу"""
lh1 = [0, 0, 0, 0]
lh2 = [0, 0, 0]
lo = [0, 0]

"""веса между входным слоем и первым скрытым"""
w_i_h1 = [
    [0.41, 0.2, 0.1, 0.23],
    [0.3, 0.3, 0.36, 0.17],
    [0.4, 0.2, 0.13, 0.05],
    [0.4, 0.2, 0.1, 0.1],
    [0.1, 0.1, 0.12, 0.35],
    [0.27, 0.3, 0.4, 0.23],
    [0.1, 0.09, 0.12, 0.12],
    [0.19, 0.21, 0.31, 0.3],
    [0.13, 0.42, 0.23, 0.02],
]

w_h1_h2 = [
    [0.34, 0.21, 0.31],
    [0.1, 0.34, 0.25],
    [0.1, 0.12, 0.36],
    [0.49, 0.2, 0.11],
]

w_h2_o = [
    [0.1, 0.3],
    [0.1, 0.1],
    [0.3, 0.1],
]

ktrain = 0.1
for x in range(0, 10000):

    print("Age number: {}".format(x))
    nset = random.randint(0, len(data_arr.lis) - 1)
    lh1 = forward(data_arr.lis[nset], lh1, w_i_h1)
    print("Train data: {}".format( data_arr.lis[nset] ))
    print("lh1: {}".format(lh1))
    lh2 = forward(lh1, lh2, w_h1_h2)
    print("lh2: {}".format(lh2))
    lo = forward(lh2, lo, w_h2_o)
    print("out: {}".format(lo))

    outerr = findErrorOut(lo, data_arr.anw[nset])
    lh2err = findError(lh2, w_h2_o, outerr)
    print("lh2 err: {}".format(lh2err))
    lh1err = findError(lh1, w_h1_h2, lh2err)
    print("lh1 err: {}" .format(lh1err))

    print("old w_h2_o")
    print(w_h2_o)

    w_h2_o = backward(lh2, lo, outerr, w_h2_o, ktrain)
    print("new w_h2_o")
    print(w_h2_o)

    print("old w_h1_h2")
    print(w_h1_h2)
    w_h1_h2 = backward(lh1, lh2, lh2err, w_h1_h2, ktrain)
    print("new w_h1_h2")
    print(w_h1_h2)

    print("old w_i_h1")
    print(w_i_h1)
    w_i_h1 = backward(data_arr.lis[nset], lh1, lh1err, w_i_h1, ktrain)
    print("new w_i_h1")
    print(w_i_h1)

for x in range(0, len(data_arr.lis)):
    print(data_arr.lis[x])
    lh1 = forward(data_arr.lis[x], lh1, w_i_h1)
    lh2 = forward(lh1, lh2, w_h1_h2)
    lo = forward(lh2, lo, w_h2_o)
    print(lo)
