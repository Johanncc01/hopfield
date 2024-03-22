import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os


class DataSaver:
    def __init__(self):
        """
        This constructor initializes a DataSaver object.
        """
        self.reset()

    def reset(self):
        """
        Resets all arguments of the DataSaver.
        """
        self.history = []
        self.energy_values = []

    def store_iter(self, state, weights):
        """
        Saves a given state to the history of the saver, and also computes and saves the energy value of the state (given the pattern's weights matrix)
        This function is to be called at each update of the network, in the dynamics method.

        Parameters
        ----------
        state : numpy array (1D)
            Computed state to be saved
        weights : numpy array (2D)
            Weight matrix of the initial state of the network
        
        Raises
        ------
        ValueError
            If the state is empty or if it contains elements other than -1 and 1.
        ValueError
            If the weights matrix is empty.
        """
        # Exeptions for the state and the weights (non-zero and binary)
        if state.size == 0:
            raise ValueError("The state must not be empty.")
        if ((state!=-1) & (state!=1)).any():
            raise ValueError("The pattern must be binary, i.e. -1 or 1.")
        if weights.size == 0:
            raise ValueError("The weights matrix must not be empty.")

        self.history.append(state)
        self.energy_values.append(self.compute_energy(state,weights))

    def compute_energy(self, state, weights):
        """
        Computes the energy associated to a specific pattern using numpy vectorization.

        Parameters
        ----------
        state : numpy array (1D)
            State which represents the current state of the network
        weights : numpy array (2D)
            Weight matrix of the initial state of the network
    
        Returns
        -------
        int
            A scalar value representing the energy of the state (non-increasing quantity)

        Raises
        ------
        ValueError
            If the state is empty or if it contains elements other than -1 and 1.
        ValueError
            If the weights matrix is empty.
        """
        # Exeptions for the state and the weights (non-zero and binary)
        if state.size == 0:
            raise ValueError("The state must not be empty.")
        if ((state!=-1) & (state!=1)).any():
            raise ValueError("The pattern must be binary, i.e. -1 or 1.")
        if weights.size == 0:
            raise ValueError("The weights matrix must not be empty.")

        return -0.5*np.dot(np.dot(state.T,weights),state).sum()

    def get_data(self): # pragma: no cover
        """
        This returns the current instance.

        Returns
        -------
        self : DataSaver
            The current instance
        """
        return self

    def save_video(self, out_path, img_shape=(50,50)):
        """
        Generates a video of the evolution of the system, provided a list of states (computed after dynamics/async).
        If the out_path folder doesn't exist, it will be created.

        Parameters
        ----------
        state_list : list
            History of states to process and export on video
        out_path : string
            Video saving path (ending with .gif or .mp4)
        img_shape :     
            Expected shape of the displayed states

        Raises
        ------
        ValueError
            If the history (of states) list is empty.
        ValueError
            If the out_path does not end with .gif or .mp4.
        """
        
        # Exceptions for the state_list (non_zero)
        if len(self.history) == 0:
            raise ValueError("The state list must not be empty.")
        
        # Exceptions for the out_path (must end with .gif or .mp4)
        if out_path.split('.')[-1] != "gif" and out_path.split('.')[-1] != "mp4":
            raise ValueError("The saving path must end with .gif or .mp4.")

        state_list = [frame.reshape(img_shape) for frame in self.history]
        images = []
        title = (out_path.split('/')[-1]).split(".")[0]
        fig, ax = plt.subplots()
        ax.set_title(title.capitalize())
        for i in range (len(state_list)):
            if i == 0 :
                ax.imshow(state_list[0], cmap='gray')
                ax.set_axis=False
            image = ax.imshow(state_list[i], cmap='gray', animated=True)
            ax.set_axis=False
            images.append([image])
        ani = animation.ArtistAnimation(fig, images)

        # Checks if the path contains a folder, and create it if it doesn't exist yet
        folder = out_path.split('/')[0]
        if folder != out_path:
            if not os.path.isdir(folder):
                os.mkdir(folder)    # pragma: no cover
        ani.save(out_path)
    
    def plot_energy(self,out_path,title='Energy from a Hopfield Network recover'):
        """
        Plots the energy values over time saved in the DataSaver, for a Hopfield network.
        Saves the plot in the given path.

        Parameters
        ----------
        out_path: str
            The path where the plot will be saved. The path must end with .jpeg or .png.
        title: str, optional
            The title of the plot. Default is 'Energy from a Hopfield Network recover'.

        Raises
        ------
        ValueError
            If the energy_values list is empty.
        ValueError
            If the out_path does not end with .jpeg or .png.
        """
        # Exceptions for energy_values and time (non_zero)
        if len(self.energy_values) == 0:
            raise ValueError("The energy values list must not be empty.")

        # Exceptions for the out_path (must end with .jpeg or .png)
        if out_path.split('.')[-1] != "jpeg" and out_path.split('.')[-1] != "png":
            raise ValueError("The saving path must end with .jpeg or .png")
        
        fig, ax = plt.subplots()
        time = list(range(len(self.history)-1))
        ax.plot(time, self.energy_values, 'g')
        ax.set_title(title)
        # Checks if the path contains a folder, and create it if it doesn't exist yet
        folder = out_path.split('/')[0]
        if folder != out_path:  # pragma: no cover
            if not os.path.isdir(folder):
                os.mkdir(folder)
        plt.savefig(out_path)
        