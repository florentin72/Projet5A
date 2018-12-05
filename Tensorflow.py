import tensorflow as tf
import numpy as np

data_path = 'data/fixture/digits.csv'
classification_path = 'data/fixture/labels.csv'
class NeuralNetwork:
  def __init__(self):
      self.hidden_layer_size = 25
      self.num_labels = 10

      self.x = None
      self.y = None
      self.decimal_labels = None
      self.samples_count = 0
      self.features_count = data_path

      self.theta1 = None
      self.theta2 = None

      self.z2 = None
      self.a2 = None
      self.z3 = None
      self.a3 = None

      self.cost = 0
      self.lambda_value = 3
      self.alpha = 1.8

  def load_samples(self, path):
      print("Loading samples : " + path)
      self.x = np.loadtxt(data_path, delimiter=",")
      self.samples_count = self.x.shape[0]
      self.features_count = self.x.shape[1]

      # Add ones to samples as first column
      #self.x = np.c_[np.ones(self.samples_count), self.x]

  def load_labels(self, path):
      print("Loading labels : " + path)
      self.decimal_labels = np.loadtxt(path, delimiter=",", dtype=np.int32)
      print(self.decimal_labels)
      # Convert decimal labels to binary vectors
      identity_matrix = np.eye(self.num_labels)
      self.y = identity_matrix[self.decimal_labels, :]

def main():
  nn = NeuralNetwork()
  nn.load_samples(data_path)
  nn.load_labels(classification_path)
  #mnist = tf.keras.datasets.mnist

  #(x_train, y_train),(x_test, y_test) = mnist.load_data()

  print (" \n \n \n ")
  print (nn.decimal_labels.shape)
  print (nn.y.shape)
  print(nn.y[1300])
  x_train = nn.x
  x_test =  nn.x
  y_test =  nn.decimal_labels
  y_train =  nn.decimal_labels

  #x_train, x_test = x_train / 255.0, x_test / 255.0

  model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
  ])
  model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

  model.fit(x_train, y_train, epochs=5)
  model.evaluate(x_test, y_test)




if __name__ == '__main__':
  main()

