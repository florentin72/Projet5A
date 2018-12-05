import time
from Layer import Layer
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold

#############################
# Global Variables          #
#############################
data_path = 'data/fixture/x.csv'
classification_path = 'data/fixture/y.csv'


def random_weights(l_in, l_out):
    """ Creates a random weight matrix."""
    epsilon_init = 0.12
    W = np.random.rand(l_out, 1 + l_in) * 2 * epsilon_init - epsilon_init
    return W


def sigmoid(z):
    """ Calculate the sigmoid of all elements in the input array."""
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_gradient(z):
    """ Calculate the sigmoid gradient of all elements in the input array."""
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))


def load_samples(path):
    """
    Returns samples from .csv file and append ones as first column.
    """
    print("Loading samples : " + path)
    x = np.loadtxt(data_path, delimiter=",")
    samples_count = x.shape[0]

    # Add ones as first column to samples
    x = np.c_[np.ones(samples_count), x]

    return x


def load_labels(path):
    """
    Returns decimal labels from .csv file.
    """
    print("Loading labels : " + path)
    return np.loadtxt(path, delimiter=",", dtype=np.int32)


def convert_y_to_bin(decimal_y, labels_count):
    """
    Convert decimal labels to binary vectors.
    """
    identity_matrix = np.eye(labels_count)
    y = identity_matrix[decimal_y, :]

    return y


def metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average=None)  # average='micro'
    precision = precision_score(y_true, y_pred, average=None)
    fmeasure = f1_score(y_true, y_pred, average=None)
    confusion_mat = confusion_matrix(y_true, y_pred)

    print("Accuracy:", accuracy)
    print()
    print("Recall:", recall)
    print()
    print("Precision:", precision)
    print()
    print("F-measure:", fmeasure)
    print()
    print("Confusion_matrix:\n", confusion_mat)
    print()


def cross_valid(x, y_true):
    print("########## Cross-validation ##########")
    nn = NeuralNetwork()

    y_true_full = []
    y_pred_full = []

    skf = StratifiedKFold(n_splits=3)
    for train_index, test_index in skf.split(x, y_true):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y_true[train_index], y_true[test_index]

        nn.initialize_weights()
        nn.train(100, x_train, y_train)
        y_pred_split = nn.predict(x_test)

        y_true_full = np.concatenate((y_true_full, y_test))
        y_pred_full = np.concatenate((y_pred_full, y_pred_split))

    metrics(y_true_full, y_pred_full)


def scalability_test(x, y_true):
    print("########## Scalability Test ##########")
    nn = NeuralNetwork()
    test_sizes = [0.8, 0.6, 0.4, 0.2]

    for test_pct in test_sizes:
        print("------ Test size {}% ------".format(test_pct))
        x_train, x_test, y_train, y_test = train_test_split(x, y_true, test_size=test_pct)

        nn.initialize_weights()
        nn.train(100, x_train, y_train)
        y_pred = nn.predict(x)

        accuracy = accuracy_score(y_true, y_pred)
        print("Accuracy:", accuracy)
        print()


def test_with_train_data(x, y_true):
    print("########## Test with Training Data ##########")
    nn = NeuralNetwork()
    nn.initialize_weights()
    nn.train(100, x, y_true)
    y_pred = nn.predict(x)
    metrics(y_true, y_pred)


class NeuralNetwork:
    def __init__(self):
        self.features_count = 50 * 50
        self.labels_count = 8
        self.hidden_layer_size = 3000
        self.nbCouches = 3
        self.theta1 = None
        self.theta2 = None
        self.theta3 = None
        self.lambda_value = 3
        self.alpha = 0.05
        self.couche =  []
        l1 = Layer(self.features_count,self.hidden_layer_size)
        # add first layer
        self.couche.append(l1)
        # add hidden layer
        for x in range(3):
            l = Layer(self.hidden_layer_size,self.hidden_layer_size)
            self.couche.append(l)
        # add output layer 
            self.couche.append(Layer(self.hidden_layer_size,self.labels_count))    


    def initialize_weights(self):
        print("Generating initial weights")
        self.theta1 = random_weights(self.features_count, self.hidden_layer_size)
        self.theta2 = random_weights(self.hidden_layer_size, 1000)
        self.theta3 = random_weights(1000, self.labels_count)
        #for l in self.couche:
         #   l.theta = random_weights(l.nbNeurone,l.sortie)

    def propagate(self, x, y):
        """
        Forward and backward propagation.
        """
        samples_count = x.shape[0]

        #############################
        # Forward propagation       #
        #############################

        # Hidden layer activation
        z2 = x @ self.theta1.transpose()
        a2 = sigmoid(z2)
        # Add ones to a2 as first column
        a2 = np.c_[np.ones(samples_count), a2]
        ######################################
        # Output layer activation
        z3 = a2 @ self.theta2.transpose()
        a3 = sigmoid(z3)
        a3 = np.c_[np.ones(samples_count), a3]
        #####################################
        # Output layer activation
        z4 = a3 @ self.theta3.transpose()
        a4 = sigmoid(z4)
       # cost = np.sum(-(y * np.log(a3) + (1 - y) * np.log(1 - a3)) / samples_count) 
        cost =  np.sum(-(y * np.log2(a4) + (1 - y) * np.log2(1 - a4)) / samples_count)
        # Add regularization
        if self.lambda_value != 0:
            sum_carre_theta1 = np.sum(self.theta1[:, 1:] ** 2)
            sum_carre_theta2 = np.sum(self.theta2[:, 1:] ** 2)
            sum_carre_theta3 = np.sum(self.theta3[:, 1:] ** 2)




            S = sum_carre_theta1 + sum_carre_theta2 +sum_carre_theta3
            cost = cost + self.lambda_value * S / (2 * samples_count)

        # print("Cost:", cost)

        #############################
        # Back propagation          #
        #############################
        delta4 = a4 - y
        delta3 = delta4 @ self.theta3[:, 1:] * sigmoid_gradient(z3)
        delta2 = delta3 @ self.theta2[:, 1:] * sigmoid_gradient(z2)

        Delta3 = delta4.transpose() @ a3
        Delta2 = delta3.transpose() @ a2
        Delta1 = delta2.transpose() @ x

        Theta1_grad = Delta1 / samples_count
        Theta2_grad = Delta2 / samples_count
        Theta3_grad = Delta3 / samples_count

        # Add regularization
        if self.lambda_value != 0:
            regularization1 = (self.lambda_value / samples_count) * self.theta1[:, 1:]
            regularization2 = (self.lambda_value / samples_count) * self.theta2[:, 1:]
            regularization3 = (self.lambda_value / samples_count) * self.theta3[:, 1:]

            Theta1_grad[:, 1:] = Theta1_grad[:, 1:] + regularization1
            Theta2_grad[:, 1:] = Theta2_grad[:, 1:] + regularization2
            Theta3_grad[:, 1:] = Theta3_grad[:, 1:] + regularization3


        # Update weights
        self.theta1 = self.theta1 - self.alpha * Theta1_grad
        self.theta2 = self.theta2 - self.alpha * Theta2_grad
        self.theta3 = self.theta3 - self.alpha * Theta3_grad

    def train(self, iteration_count, x, decimal_y):
        start_time = time.time()
        y = convert_y_to_bin(decimal_y, self.labels_count)
        for i in range(iteration_count):
            self.propagate(x, y)

        end_time = time.time() - start_time
        print("Training execution time: {} seconds".format(end_time))

    def predict(self, x):
        start_time = time.time()

        samples_count = x.shape[0]

        # Hidden layer activation
        z2 = x @ self.theta1.transpose()
        a2 = sigmoid(z2)

        # Add ones to a2 as first column
        a2 = np.c_[np.ones(samples_count), a2]

        # 2nd layer hidden
        z3 = a2 @ self.theta2.transpose()
        a3 = sigmoid(z3)

        # Add ones to a3 as first column
        a3 = np.c_[np.ones(samples_count), a3]       
        # Output 
        z4 = a3 @ self.theta3.transpose()
        a4 = sigmoid(z4)
        # Convert binary prediction to de
        decimal_prediction = a4.argmax(1)
        end_time = time.time() - start_time
        print("Prediction execution time:")
        return decimal_prediction


def main():
    x = load_samples(data_path)
    y = load_labels(classification_path)

    test_with_train_data(x, y)
    scalability_test(x, y)
    cross_valid(x, y)

    print("END")


if __name__ == '__main__':
    main()
