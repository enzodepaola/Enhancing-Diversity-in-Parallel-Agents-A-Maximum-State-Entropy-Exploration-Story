import numpy as np


class SoftMaxPolicy:

    def __init__(self, agents,action_dim,obs_dim,test_probs,zero_init,single_agent=False,std=0.1):
        """
        :param dim: number of state variables
        :param std: fixed standard deviation
        """

        self.std = std
        self.agents = agents
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.single_agent = single_agent

        if zero_init:
            if single_agent:
                self.theta = np.zeros((1,action_dim,obs_dim))
            else:
                self.theta = np.zeros((agents,action_dim,obs_dim)) 
        else:
            if single_agent:
                 self.theta =2*np.random.rand(1,action_dim, obs_dim) -1
            else:
                 self.theta =2*np.random.rand(agents,action_dim, obs_dim) -1
             

        #self.theta = np.zeros((agents,action_dim,obs_dim))  # zero initializatoin #np.array([[[-0.4,-2,-0.25],[0.4,2,0.25]],[[4,-2,4],[-4,2,-4]]])
        self.epsilon = 0.1
        self.force_probs = False
        self.test_probs = test_probs
        
        self.action_rng = np.random.default_rng()

    def get_theta(self):
        return self.theta

    def set_theta(self, value):
        ### Need to verify that theta is a matrix with action_dim,obs_dim
        self.theta = value
    
    def set_force_probs(self, value):
        self.force_probs = value

    def predict(self, obs):
        probs  = self.get_action_probabilities(obs)
        if self.force_probs:
            action = [self.action_rng.choice(self.action_dim,p=self.test_probs) for index in range(len(probs))]#self.action_rng.choice(self.action_dim,p=self.test_probs)
        else:
            action = [self.action_rng.choice(self.action_dim,p=probs[index]) for index in range(len(probs))]
        return np.array([action]).reshape(len(probs),)
    
        # Function to compute softmax
    def softmax(self,logits):
        exp_logits = np.exp(logits) # Subtract max for numerical stability
        probs =  exp_logits / np.sum(exp_logits,axis=1)[:, None] 
        # Assuming probs is a 2D numpy array where each row needs to sum to 1
        for i in range(len(probs)):
            difference = 1 - np.sum(probs[i])
            if difference > 0:
               probs[i,np.argwhere(probs[i] == np.min(probs[i]))[0]] += difference
            elif difference < 0:
                probs[i,np.argwhere(probs[i] == np.max(probs[i]))[0]] += difference
        #print(difference)
        return probs

    # Function to get action probabilities for a given state
    def get_action_probabilities(self,state):
        logits = np.einsum('ijk,ik->ij', self.theta, state)
        return self.softmax(logits)
    
    def log_softmax_derivative(self,s,a):
        """Calcola la matrice Jacobiana della log-softmax."""
        grad = []
        x = np.einsum('ijk,ik->ij', self.theta, s)
        fun = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)  # keepdims per mantenere la dimensione
        for agent,action in enumerate(a):
            grad.append([np.dot((1 - fun[agent,i]),s[agent]) if i == action else -np.dot(fun[agent,i],s[agent]) for i in range(self.action_dim)])
       
       #grad = [np.dot((1 - fun[i]),s) if i == a else -np.dot(fun[i],s) for i in range(self.action_dim)]

        return grad


class SoftMax_nst_Policy:

    def __init__(self, agents,action_dim,obs_dim,horizon,test_probs,zero_init,std=0.1):
        """
        :param dim: number of state variables
        :param std: fixed standard deviation
        """

        self.std = std
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.agents = agents
        #generate random theta from -1 to 1
        if zero_init:
            self.theta = np.zeros((agents,horizon,action_dim,obs_dim)) 
        else:
            self.theta =2*np.random.rand(agents, horizon, action_dim, obs_dim) -1      
        
        self.epsilon = 0.1
        self.force_probs = False
        self.test_probs = test_probs
        self.action_rng = np.random.default_rng()

    def get_theta(self):
        return self.theta

    def set_theta(self, value):
        ### Need to verify that theta is a matrix with action_dim,obs_dim
        self.theta = value
    
    def set_force_probs(self, value):
        self.force_probs = value

    def predict(self, obs,time):
        probs  = self.get_action_probabilities(obs,time)
        if self.force_probs:
            #action = [np.searchsorted(np.cumsum(probs[index]), np.random.rand()) for index in range(len(probs))]
            action = [self.action_rng.choice(self.action_dim,p=self.test_probs) for index in range(len(probs))]#self.action_rng.choice(self.action_dim,p=self.test_probs)
        else:
            #action = [np.searchsorted(np.cumsum(probs[index]), np.random.rand()) for index in range(len(probs))] ##Cupy version
            action = [self.action_rng.choice(self.action_dim,p=probs[index]) for index in range(len(probs))]
        return np.array([action]).reshape(len(probs),)
    
        # Function to compute softmax
    def softmax(self,logits):
        exp_logits = np.exp(logits) # Subtract max for numerical stability
        probs =  exp_logits / np.sum(exp_logits,axis=1)[:, None] 
        # Assuming probs is a 2D numpy array where each row needs to sum to 1
        for i in range(len(probs)):
            difference = 1 - np.sum(probs[i])
            if difference > 0:
               probs[i,np.argwhere(probs[i] == np.min(probs[i]))[0]] += difference
            elif difference < 0:
                probs[i,np.argwhere(probs[i] == np.max(probs[i]))[0]] += difference
        #print(difference)
        return probs

    # Function to get action probabilities for a given state
    def get_action_probabilities(self,state,time):
        logits = np.einsum('ijk,ik->ij', self.theta[:,time,:,:], state)
        return self.softmax(logits)
    
    def log_softmax_derivative(self,s,a,time):
        """Calcola la matrice Jacobiana della log-softmax."""
        grad = []
        x = np.einsum('ijk,ik->ij', self.theta[:,time,:,:], s)
        fun = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)  # keepdims per mantenere la dimensione
        for agent,action in enumerate(a):
            grad.append([np.dot((1 - fun[agent,i]),s[agent]) if i == action else -np.dot(fun[agent,i],s[agent]) for i in range(self.action_dim)])
            
       #grad = [np.dot((1 - fun[i]),s) if i == a else -np.dot(fun[i],s) for i in range(self.action_dim)]

        return grad


class LinearPolicy:
    def __init__(self, agents,action_dim,obs_dim,horizon,zero_init,std=0.1):
        """
        :param dim: number of state variables
        :param std: fixed standard deviation
        """

        self.std = std
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.agents = agents

        if zero_init:
            self.theta = np.zeros((agents,horizon,obs_dim)) 
        else:
            self.theta = np.random.rand(agents, horizon, obs_dim)      
        
        self.epsilon = 0.1
        self.force_probs = False
        self.action_rng = np.random.default_rng()


    def get_theta(self):
        return self.theta

    def set_theta(self, value):
        ### Need to verify that theta is a matrix with action_dim,obs_dim
        self.theta = value
    
    def set_force_probs(self, value):
        self.force_probs = value
    
    def get_action_probabilities(self,state,timestep):
        prob = np.sum(np.dot(self.theta[:, timestep], state),axis=-1)
        prob = np.clip(prob, 0, 1)  # Clamping
        return prob

    def predict(self, obs,time):
        action = np.zeros(self.agents)
        probs  = self.get_action_probabilities(obs,time)
        for index in range(len(probs)):
            action[index] = 0 if np.random.rand() < probs[index] else 1
        return action
    
    def log_softmax_derivative(self,s,a,time):
        """Calcola la matrice Jacobiana della log-softmax."""
        grad = []
        x = np.dot(self.theta[:,time], s)

        for agent,action in enumerate(a):
            if action == 0:
                grad.append(1/x[agent])
            else:
                grad.append(-1/(1-x[agent]) * s[agent])
        return grad