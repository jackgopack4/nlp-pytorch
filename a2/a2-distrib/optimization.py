# optimization.py

import argparse
import numpy as np
import scipy.special
import matplotlib.pyplot as plt

### 
# IMPLEMENT ME! REPLACE WITH YOUR ANSWER TO PART 1B
OPTIMAL_STEP_SIZE = 0.105 # Any value between 0.103216 and 0.108981 will result in 10 epochs
###

def _parse_args():
    """
    Command-line arguments to the system.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='optimization.py')
    parser.add_argument('--func', type=str, default='QUAD', help='function to optimize (QUAD or NN)')
    parser.add_argument('--lr', type=float, default=OPTIMAL_STEP_SIZE, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0., help='weight decay')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    args = parser.parse_args()
    return args

def point_distance(p1,p2):
    # calculates distance between two points
    # take in two numpy arrays
    return np.sqrt(np.square(p1[0]-p2[0])+np.square(p1[1]-p2[1]))

def quadratic(x1, x2):
    """
    Quadratic function of two variables
    :param x1: first coordinate
    :param x2: second coordinate
    :return:
    """
    return (x1 - 1) ** 2 + 8 * (x2 - 1) ** 2


def quadratic_grad(x1, x2):
    """
    Should return a numpy array containing the gradient of the quadratic function defined above evaluated at the point
    :param x1: first coordinate
    :param x2: second coordinate
    :return: a two-dimensional numpy array containing the gradient
    """
    g1 = 2*(x1-1)
    g2 = 16*(x2-1)
    return np.array([g1,g2])
    #raise Exception("Implement me!")


def sgd_test_quadratic(args):
    xlist = np.linspace(-3.0, 3.0, 100)
    ylist = np.linspace(-3.0, 3.0, 100)
    X, Y = np.meshgrid(xlist, ylist)
    Z = quadratic(X, Y)
    plt.figure()

    # Track the points visited here
    points_history = []
    curr_point = np.array([0, 0])
    for iter in range(0, args.epochs):
        grad = quadratic_grad(curr_point[0], curr_point[1])
        if len(grad) != 2:
            raise Exception("Gradient must be a two-dimensional array (vector containing [df/dx1, df/dx2])")
        next_point = curr_point - args.lr * grad
        points_history.append(curr_point)
        difference = point_distance(np.array([1,1]),next_point)
        print("Point after epoch %i: %s, distance: %f" % (iter+1, repr(next_point),difference))
        #if(difference<0.1):
        #    break
        curr_point = next_point
    points_history.append(curr_point)
    cp = plt.contourf(X, Y, Z)
    plt.colorbar(cp)
    plt.plot([p[0] for p in points_history], [p[1] for p in points_history], color='k', linestyle='-', linewidth=1, marker=".")
    plt.title('SGD on quadratic for lr %f'%args.lr)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    exit()


if __name__ == '__main__':
    args = _parse_args()
    sgd_test_quadratic(args)
