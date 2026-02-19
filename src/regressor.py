import numpy as np

class LogisticRegressor:
    def __init__(self, 
                 batch_size=1,
                 learning_rate=0.01,
                 num_epochs=100,
                 regularization_strength=0):
        
        self.batch_size = batch_size
        self.learing_rate=learning_rate
        self.num_epochs=num_epochs
        self.regularization_strength = regularization_strength
    
    def test(self):
        return self.batch_size
    
    def fit(self, learning_curve=False):
        return np.arange(1, self.num_epochs+1) ** 1