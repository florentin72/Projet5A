from numpy import array
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.metrics import classification_report, f1_score, roc_curve, confusion_matrix
import datetime

#############################
# Global Variables          #
#############################
data_path = 'data/fixture/x.csv'
classification_path = 'data/fixture/y.csv'

# Cross Validation
# from sklearn.model_selection  import cross_val_score, train_test_split
# # Split full data into training and testing subsets.
# X_train, X_test, y_train, y_test = train_test_split(self.data, self.classifications, test_size=0.33)

class SKLearnModels:
    def __init__(self):
        self.data = []
        self.classifications = []
        self.load_data()

    def load_data(self):
        print("Loading data from CSV file : " + data_path)
        self.data = np.loadtxt(data_path, delimiter=",")
        self.classifications = np.loadtxt(classification_path, delimiter=",")

    def run(self, classifier, modelName):

        # Train model with all dataset but last sample, used later for prediction.
        training_start_time = datetime.datetime.now()
        classifier.fit(self.data[:-1], self.classifications[:-1])
        training_end_time = datetime.datetime.now()

        # Predict the class of the last sample
        prediction_start_time = datetime.datetime.now()
        predictions = classifier.predict(self.data)
        prediction_end_time = datetime.datetime.now()

        print("============== " + str(modelName) + " Performance reports ==============\n\r")
        target_names = ['Cercles_2_F', 'Cercles_3_F', 'Cercles_4_F', 'Hexagones_2_F', 'Hexagones_3_F', 'Losanges_2_F', 'Losanges_3_F', 'Triangles_2_F']
        class_report = classification_report(self.classifications, predictions, target_names=target_names)
        print(class_report)
        print("----------------- Confusion Matrix ------------------\n\r")
        confusion_matrix_array = confusion_matrix(self.classifications, predictions)
        print(str(confusion_matrix_array) + "\n\r")
        print("------------------ Execution Time -------------------\n\r")
        print("Training time   :  " + str(training_end_time - training_start_time))
        print("Prediction time :  " + str(prediction_end_time - prediction_start_time) + "\n\r")
        print("=====================================================\n\r")

def main():

    classifier_knn = classifier = KNeighborsClassifier(n_neighbors=3)
    classifier_rn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    classifier_svm = svm.SVC(gamma=0.001, C=100.)

    skl = SKLearnModels()
    skl.run(classifier_knn, "KNN")
    skl.run(classifier_rn, "RN")
    skl.run(classifier_svm, "SVM")

if __name__ == '__main__':
    main()