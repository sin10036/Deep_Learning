import numpy as np
import matplotlib.pyplot as plt

from models.neural_net import NeuralNetwork

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython

def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


# Create a small net and some toy data to check your implementations.
# Note that we set the random seed for repeatable experiments.

input_size = 4
hidden_size = 10
num_classes = 3
num_inputs = 5


def init_toy_model(nonlinearity, num_layers):
    np.random.seed(0)
    return NeuralNetwork(input_size, [hidden_size]*(num_layers-1), num_classes, num_layers, nonlinearity=nonlinearity)

def init_toy_data():
    np.random.seed(1)

    X = 10 * np.random.randn(num_inputs, input_size)
    y = np.array([0, 1, 2, 2, 1])
    return X, y

from utils.gradient_check import eval_numerical_gradient

X, y = init_toy_data()

# Use numeric gradient checking to check your implementation of the backward pass.
# If your implementation is correct, the difference between the numeric and
# analytic gradients should be around 1e-8 or less for each of the parameters 
# W1, b1,... in your network.

for net_activation in ['sigmoid', 'relu']:
    for num in [2, 3]:
        print(net_activation)
        net = init_toy_model(net_activation, num)
        loss, grads = net.loss(X, y, reg=0.05)
        print(loss)

        # these should all be less than 1e-8 or so
        for param_name in grads:
            f = lambda W: net.loss(X, y, reg=0.05)[0]
            param_grad_num = eval_numerical_gradient(f, net.params[param_name], verbose=False)
            print('%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name])))

net = init_toy_model('relu', 2)
stats = net.train(X, y, X, y,
            learning_rate=1e-1, reg=5e-6,
            num_iters=100, verbose=False)

print('Final training loss: ', stats['loss_history'][-1])

# plot the loss history
plt.plot(stats['loss_history'])
plt.xlabel('iteration')
plt.ylabel('training loss')
plt.title('Training Loss history')
plt.show()




