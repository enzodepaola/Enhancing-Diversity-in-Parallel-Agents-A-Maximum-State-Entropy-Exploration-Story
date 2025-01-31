import numpy as np
import math

class AdamOptimizer:
    def __init__(self, theta, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # Initialize first and second moments (for Adam)
        self.m = np.zeros_like(theta)  # First moment vector
        self.v = np.zeros_like(theta)  # Second moment vector
        self.t = 0  # Time step

    def update(self, grads):
        """
        Applies Adam update rule
        :param policy: (Policy object) the policy to be updated
        :param grads: (np.array) the gradients for the current policy
        """
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)

        # Bias-corrected first and second moment estimates
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # Update policy parameters (theta)
        return self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


class LearningRateScheduler:
    def __init__(self, initial_lr, decay_rate, decay_steps):
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

    def get_lr(self, epoch):
        lr = self.initial_lr * math.exp(-self.decay_rate * epoch / self.decay_steps)
        return lr
    