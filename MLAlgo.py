'''
Wrapper for all MachineLearning algorithm
'''

from sklearn.neighbors import KNeighborsClassifier

class KNN:
    def __init__(self, n_neighbors):
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    def train(self, X_train, y_train):
        self.knn.fit(X_train, y_train)
        
    def score(self, X_test, y_test):
        return self.knn.score(X_test, y_test)
    
    def predict(self, X_test):
        return self.knn.predict(X_test)