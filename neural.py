from math import exp
import numpy


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
li = [1.0, 0.5]

"""скрытый слой 1"""
"""перейти за заполнение массива сразу"""
lh1 = [0, 0]
lo = [0, 0]

"""веса между входным слоем и первым скрытым"""
w_i_h1 = [
    [0.9, 0.2],
    [0.3, 0.8]
]

w_h1_o = [
    [0.3, 0.6],
    [0.7, 0.5],
]

lh1 = forward(li, lh1, w_i_h1)
lo = forward(lh1, lo, w_h1_o)
print(lh1)
print(lo)
