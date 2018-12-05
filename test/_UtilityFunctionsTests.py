from NeuralNetwork import random_weights, sigmoid, sigmoid_gradient
import numpy as np
def main():
    print("SIGMOID ==========================================")
    print("Sigmoid(-5): " + str(sigmoid(-5)))
    print("Sigmoid(0): " + str(sigmoid(0)))
    print ("Sigmoid Gradient  W = np.random.rand(20,5)") 
    W = np.random.rand(20,5)
    print (sigmoid_gradient(W))
    print("RANDINITIALIZEWEIGHT =============================")
    print("RandInitializeWeight(2,2): ")
    print(str(random_weights(2, 2)))
    print("RandInitializeWeight(10,20): ")
    print(str(random_weights(10, 20)))


if __name__ == '__main__':
    main()
