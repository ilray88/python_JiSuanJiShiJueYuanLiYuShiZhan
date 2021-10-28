import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-10, 10, num=1000)

def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y

def tanh(x):
    return 2 * sigmoid(x) - 1

def relu(x):
    y = [0 if _x <= 0 else _x for _x in x]
    return np.asarray(y)

def leaky_relu(x, alpha=0.2):
    y = [alpha * _x if _x <= 0 else _x for _x in x]
    return np.asarray(y)

def relu6(x):
    y = np.minimum(np.maximum(0, x), 6)
    return y

def elu(x, alpha):
    def f(x, alpha):
        return alpha * (np.exp(x) - 1)
    y = [f(_x, alpha) if _x < 0 else _x for _x in x]
    return np.asarray(y)

def swish(x, beta=1):
    y = [_x * sigmoid(_x) for _x in x]
    return np.asarray(y)

def mish(x):
    y = [_x * tanh(np.log(1 + np.exp(_x))) for _x in x]
    return np.asarray(y)

def draw(x, y, name):
    plt.plot(x, y)
    plt.grid()
    plt.savefig(name)
    plt.show()

if __name__ == "__main__":
    # sig = sigmoid(x)
    # draw(x, sig, 'sigmoid.jpg')

    # tan = tanh(x)
    # draw(x, tan, 'tanh.jpg')

    # rel = relu(x)
    # draw(x, rel, 'relu.jpg')

    # lrel = leaky_relu(x)
    # draw(x, lrel, 'lrelu.jpg')

    # rel6 = relu6(x)
    # draw(x, rel6, 'relu_6.jpg')
    
    # el = elu(x, alpha=0.2)
    # draw(x, el, 'elu.jpg')

    # sws = swish(x)
    # draw(x, sws, 'swish.jpg')

    mis = mish(x)
    draw(x, mis, 'mish.jpg')

    pass