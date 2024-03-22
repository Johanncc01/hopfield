from patterns import *

class HopfieldNetwork:
    def __init__(self, patterns, rule="hebbian"):
        """
        This constructor initializes a HopefieldNetwork object from patterns and a learning rule, which specifies how the network's weights should be computed.

        Parameters
        ----------
        patterns : numpy array (2D)
            Patterns to be stored in the network
        rule : str, optional
            The update rule used to initialize the patterns. Valid values are "hebbian" and "storkey" (by default "hebbian").

        Raises
        ------
        ValueError
            If the patterns are empty or if it contains elements other than -1 and 1.
        """

        if patterns.size == 0:
            raise ValueError("The patterns must not be empty.")
        if ((patterns!=-1) & (patterns!=1)).any():
            raise ValueError("The patterns must be binary, i.e. -1 or 1.")

        if rule == "hebbian" or rule == "Hebbian":
            self.weights = self.hebbian_weights(patterns)
        elif rule == "storkey" or rule == "Storkey" :
            self.weights = self.storkey_weights(patterns)
        else :
            raise ValueError("The rule should be either hebbian or storkey.")

    def hebbian_weights(self, patterns):
        """
        Applies the Hebbian learning rule on a given set of patterns using numpy vectorization to efficiently create the weight matrix.

        Parameters
        ----------
        patterns : numpy array (2D)
            Memorized patterns to process

        Returns
        -------
        numpy array (2D)
            Weight matrix (of shape : (pattern_size, pattern_size)) corresponding to the patterns.

        Raises
        ------
        ValueError
            If the patterns are empty.

        Example
        -------
        >>> network = HopfieldNetwork(np.array([[1,1,-1,-1],[1,1,-1,1],[-1,1,-1,1]]), "hebbian")
        >>> network.weights
        array([[ 0.        ,  0.33333333, -0.33333333, -0.33333333],
               [ 0.33333333,  0.        , -1.        ,  0.33333333],
               [-0.33333333, -1.        ,  0.        , -0.33333333],
               [-0.33333333,  0.33333333, -0.33333333,  0.        ]])
        """       
        if patterns.size == 0 :
            raise ValueError("The patterns must not be empty.")

        weights = (1/len(patterns) * np.dot(patterns.T, patterns))
        np.fill_diagonal(weights, 0)
        return weights

    def storkey_weights(self, patterns):
        """
        Applies the Storkey learning rule on a given set of patterns using numpy vectorization to efficiently create the weight matrix.

        Parameters
        ----------
        patterns : numpy array (2D)
            Memorized patterns to process

        Returns
        -------
        numpy array (2D)
            Weight matrix (of shape : (pattern_size, pattern_size)) corresponding to the patterns.

        Raises
        ------
        ValueError
            If the patterns are empty.
        
        Example
        -------
        >>> network = HopfieldNetwork(np.array([[1,1,-1,-1],[1,1,-1,1],[-1,1,-1,1]]), "storkey")
        >>> network.weights
        array([[ 1.125,  0.25 , -0.25 , -0.5  ],
               [ 0.25 ,  0.625, -1.   ,  0.25 ],
               [-0.25 , -1.   ,  0.625, -0.25 ],
               [-0.5  ,  0.25 , -0.25 ,  1.125]])
        """
        if patterns.size == 0:
            raise ValueError("The patterns must not be empty.")

        M = len(patterns)
        N = len(patterns[0])
        weights = np.zeros((N,N))
        for k in range(M):
            H = np.dot(weights, patterns[k].reshape(N,1)) + np.zeros(N) - np.diag(weights)*np.reshape(patterns[k],(N,1)) -weights*patterns[k] + np.diag(np.diag(weights*patterns[k]))
            weights = weights + 1/N*(np.reshape(patterns[k],(N,1))*patterns[k] -H*patterns[k] - np.transpose(H)*np.reshape(patterns[k],(N,1)))
        return weights

    def update(self, state):
        """
        Applies the update rule to a state pattern.

        Parameters
        ----------
        state : numpy array (1D)
            Describes the network state
        weights : numpy array (2D)
            Weight matrix of the starting patterns

        Returns
        -------
        numpy array (1D)
            New network state computed

        Raises
        ------
        ValueError
            If the patterns or the weights are empty.
        """
        if state.size == 0 or self.weights.size == 0:
            raise ValueError("The state or the weights must not be empty")

        return np.where(np.dot(self.weights, state) < 0, -1, 1)

    def update_async(self, state):
        """
        Applies the asynchronous update rule to a state pattern, i.e. updates the i-th componant of the state vector (i sampled uniformly at random)

        Parameters
        ----------
        state : numpy array (1D)
            Describes the network state
        weights : numpy array (2D)
            Weight matrix of the initial state of the network

        Returns
        -------
        numpy array (1D)
            New network state computed

        Raises
        ------
        ValueError
            If the patterns or the weights are empty.
        """
        if state.size == 0 or self.weights.size == 0:
            raise ValueError("The state or the weights must not be empty")

        i = np.random.randint(len(self.weights))
        new_state = np.copy(state)
        new_state[i] = np.where(np.dot(self.weights[i], state) < 0, -1, 1)
        return new_state 

    def dynamics(self, state, saver, max_iter=20):
        """
        Run the dynamical system from an initial state until convergence (two consecutives updates return the same state), or until a given nuber of states.

        Parameters
        ----------
        state : numpy array (1D)
            Initial state of the network
        weights : numpy array (2D)
            Weight matrix of the initial state of the network
        max_iter : int
            Number of maximum iteration until the end of the execution

        Returns
        -------
        list
            Whole state history containing each update.

        Raises
        ------
        ValueError
            If the initial state is empty or if it contains elements other than -1 and 1.
        ValueError
            If the maximal number of iterations is not a positive integer.
        """
        # Exceptions for the state (non-zero and binary)
        if state.size == 0:
            raise ValueError("The initial state must not be empty.")
        if ((state!=-1) & (state!=1)).any():
            raise ValueError("The pattern must be binary, i.e. -1 or 1.")

        # Exceptions for max_iter (positive integers)
        if max_iter <= 0:
            raise ValueError("The maximal number of iterations has to be positive.")
        if not(float(max_iter).is_integer()):
            raise ValueError("The number of perturbations must be an integer.")

        saver.history = [state]
        t = 0
        old_state = np.copy(state)
        while t < max_iter:
            new_state = self.update(old_state)
            saver.store_iter(new_state, self.weights)

            t+=1
            if ((new_state == old_state).all()):
                break
            old_state = np.copy(new_state)

    def dynamics_async(self, state, saver, max_iter=1000, convergence_num_iter=100, skip=10):
        """
        Run the dynamical system (with asynchronous updates) from an initial state until convergence (many consecutives updates return the same state), or until a given nuber of states.

        Parameters
        ----------
        state : numpy array (1D)
            Initial state of the network
        weights : numpy array (2D)
            Weight matrix of the initial state of the networka
        max_iter : int
            Number of maximum iteration until the end of the execution
        convergence_num_iter : int
            Number of consecutives updates required to reach convergence
        skip : int
            Number of iterations to skip between each save of the evolution

        Returns
        -------
        list
            Whole state history containing each asynchronous update.

        Raises
        ------
        ValueError
            If the initial state is empty or if it contains elements other than -1 and 1.
        ValueError
            If either the maximal number of iterations, the convergence number or the skip number is not a positive integer.
        ValueError
            If the convergence number of iterations is greater than the maximal number of iterations.
        """
        # Exeptions for the state (non-zero and binary)
        if state.size == 0:
            raise ValueError("The initial state must not be empty.")
        if ((state!=-1) & (state!=1)).any():
            raise ValueError("The pattern must be binary, i.e. -1 or 1.")

        # Exceptions for max_iter, convergence_max_iter and skip(positive integers)
        if max_iter <= 0 or convergence_num_iter <= 0 or skip <= 0:
            raise ValueError("Maximal iterations values have to be positive.")
        if not(float(max_iter).is_integer()) and not(float(convergence_num_iter).is_integer()) and not(float(skip).is_integer()):
            raise ValueError("Maximal iterations values must be integers.")
        if convergence_num_iter > max_iter:
            raise ValueError("Convergence max value must be lower than global max value.")

        saver.history = [state]
        t = 0
        stable_steps = 0
        old_state = np.copy(state)
        while t < max_iter:
            new_state = self.update_async(old_state)
            if t%skip==0 :
                saver.store_iter(new_state, self.weights)
            t+=1

            if ((new_state == old_state).all()):
                stable_steps+=1
            else:
                stable_steps = 0
            old_state = np.copy(new_state)

            if (stable_steps >= convergence_num_iter):
                break

if __name__ == "__main__":  # pragma: no cover
    import doctest # Importing the library
    print("Starting doctests") # not required (just for clarity in output)
    doctest.testmod() 
