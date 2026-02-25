import numpy as np
import itertools

# let y^(i)_hat \in [0, 1] be our prediction (a probability)
# that x^{(i)} is a yes-instance of the class.

# For our logistic classifier, y^{(i)}_hat := \sigma(w^T x^{(i)}) 

# The likelihood of our model classifying x^(i) correctly is 
# the Bernoulli(y^{(i)} ; y^(i)_hat)
# = (\hat{y}^{(i)})^{y^{(i)}} * (1 - \hat{y}^{(i)})^{1 - y^{(i)}}

# Thus the negative log likelihood is then 
# - y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)})\log(1 - \hat{y}^{(i)})
    
# This is the objective funciton of our logistic classifier.
def cross_entropy_loss(pred, y):
        return ( -y * np.log(pred) - (1 - y) * np.log(1 - pred) ).sum()


class LogisticRegressor:
    def __init__(self, 
                 batch_size=1,
                 learning_rate=0.01,
                 num_epochs=100,
                 regularization_strength=0,
                 seed=None):
        
        self.batch_size = batch_size
        self.learing_rate=learning_rate
        self.num_epochs=num_epochs
        self.regularization_strength = regularization_strength

        self.rng = np.random.default_rng(seed)
    
    # TODO: implement ADAM
    def fit(self, X, y, learning_curve=False):
        # Concatenate 1's for bias terms
        X = np.concatenate([np.ones(X.shape[0]), X], axis=1)

        # Add one for bias 
        self.weights = self.rng.random(X.shape[0] + 1)

        if(learning_curve):
            errors = [self.loss(self.predict(X), y)]

        for epoch_count in range(self.num_epochs):
            # Permute our rows for a new epoch
            X_p = self.rng.permute(X)

            for batch in itertools.batched(X_p, self.batch_size):
                self.weights = ( 
                    self.weights -
                    self.learning_rate * self.loss_gradient(batch, y) 
                )

            if(learning_curve):
                errors.append(self.loss(X, y))

        return errors
    
    # Our loss is the L2-regularized cross entropy loss
    def loss(self, X, y):
        return ( 
            cross_entropy_loss(self.predict(X), y) +
                (self.regularization_strength / 2.0) * (
                        np.linalg.norm(self.weights) ** 2 - self.weights[0] # no bias penalty
                    )
        )
    
    # The gradient of our loss function
    def loss_gradient(self, X, y):
        return ( 
            np.sum(X * (self.predict(X) - y)) + 
            self.regularization_strength * self.weights
        )
    
    # prediction of logistic classifer use the sigmoid function
    def predict(self, x):
        return 1.0 / (1 - np.exp(-self.weights.transpose() @ x))