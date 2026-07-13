import numpy as np

def linear(x):
    return x

#sqrt: np.sqrt(pos)
def square_root(x):
    return np.sqrt(x)

#4th root: np.pow(pos, 0.25)
def fourth_root(x):
    return np.power(x,0.25)

#8th root: np.pow(pos, 0.125)
def eighth_root(x):
    return np.power(x, 0.125)

#16th root: np.pow(pos, 0.0625)
def sixteenth_root(x):
    return np.power(x, 0.0625)

#exp: np.exp(pos - n) - np.exp(-n); n = np.log(np.e - 1)
def exponential(x):
    n = np.log(np.e - 1)
    return np.exp(x - n) - np.exp(-n)

#10^: np.pow(10, pos - n) - np.pow(10,-n); n = np.log10(10 - 1)
def exponential_ten(x):
    n = np.log10(10 - 1)
    return np.power(10, x - n) - np.power(10,-n)

#sin: np.sin(np.pi * pos * 0.5)
def sine(x):
    return np.sin(np.pi * x * 0.5)

#tan: 0.09 * np.tan(2.60*pos + 1.82) + 0.52; this one is pretty funky ngl
def tangent(x):
    return 0.09 * np.tan(2.60*x + 1.82) + 0.52

#sigmoid: 1 / (1 + np.exp(-8 * (pos - 0.5))); not full range / is adjustable
def sigmoid(x):
    return 1 / (1 + np.exp(-8 * (x - 0.5)))

#jet sigmoid: 1 / (1 + np.exp(-6 * (pos - 0.5)))
def sigmoid2(x):
    return 1 / (1 + np.exp(-6 * (x - 0.5)))

#flip: 1 - (any mapping)
def flip(f,x):
    return 1 - f(x)



MAPPINGS = {"linear" : linear,
            "sqrt" : square_root,
            "4th_root": fourth_root,
            "8th_root": eighth_root,
            "16th_root": sixteenth_root,
            "e^x" : exponential,
            "10^x" : exponential_ten,
            "sin" : sine,
            "tan" : tangent,
            "sigmoid" : sigmoid,
            "sigmoid2" : sigmoid2,
            "flip" : flip} #stores mapping functions