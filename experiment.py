from HopfieldNetwork import *
from DataSaver import *
import random
import pandas as pd

                                       # Experiment Functions

def capacity(n, weight_rule):
    """
    Calculates the theoretical capacity of a Hopfield network with a given number
    of neurons and weight rule.
    
    Parameters
    ----------
    n : int
        The number of neurons in the network.
    weight_rule : str 
        The weight rule to use for the network, either "hebbian" or "storkey".
    
    Returns
    -------
    float
        The theoretical capacity of the network.
    
    Raises
    ------
    ValueError:
    If the number of neurons is not a strictly positive integer, or if the weight rule is not "hebbian" or "storkey".
    
    Example
    -------
    >>> capacity(10, "hebbian")
    2.1714724095162588
        
    >>> capacity(50, "storkey")
    17.875339809194237
    """

    # ensure that the number of neurons is an integer and strictly positive
    if n <= 1:
        raise ValueError('The number of neurons needs to be strictly positive')
    if int(n) != n:
        raise ValueError('The number of neurons needs to be an integer')

    # ensure that the weight rule is either hebbian or storkey
    if weight_rule != 'hebbian' and weight_rule != 'storkey' :
        raise Exception('The weight rule needs to be either hebbian or storkey')

    if weight_rule == "hebbian":
        return n/(2*log(n))
    elif weight_rule == "storkey":
        return n/(sqrt(2*log(n)))

def num_patterns_generator(n, weight_rule):
    """
    Generates a list of possible number of patterns to use in a capacity
    experiment for a Hopfield network with a given number of neurons and
    weight rule.
    
    
    Parameters
    ----------
    n : int
        The number of neurons in the network.
    weight_rule : str
        The weight rule to use for the network, either "hebbian" or "storkey".
    
    Returns
    -------
    list[int]
        A sorted list of possible number of patterns to use in the capacity experiment.
    
    Example
    -------
    >>> num_patterns_generator(10, "hebbian")
    [1, 2, 3, 4]
        
    >>> num_patterns_generator(50, "storkey")
    [8, 11, 14, 17, 20, 23, 26, 29, 32, 35]
    """
    return sorted(set(np.linspace(0.5 * capacity(n, weight_rule), 2 * capacity(n, weight_rule), 10).astype(int)))

def experiment(size, num_patterns, weight_rule, num_perturb = 0.2, num_trials=100, max_iter=100):
    """
    Runs a capacity experiment for a Hopfield network with a given size, number
    of patterns, weight rule, number of perturbations, number of trials, and
    maximum number of iterations.
    
    Parameters
    ----------
    size : int
        The size of the network.
    num_patterns : int 
        The number of patterns to use in the experiment.
    weight_rule : str
        The weight rule to use for the network, either "hebbian" or "storkey".
    num_perturb : float
        The fraction of neurons to perturb in each trial.
    num_trials : int
        The number of trials to run.
    max_iter : int
        The maximum number of iterations to run in each trial.
    
    Returns
    -------
    dict
        A dictionary containing the results of the experiment.
    """
    
    saver = DataSaver()
    patterns = generate_patterns(num_patterns, size)
    network = HopfieldNetwork(patterns, weight_rule)
    retrieved_patterns = 0

    for j in range(num_trials):
        index = random.choice(range(num_patterns))
        perturbated = perturb_pattern(patterns[index], floor(num_perturb*size))
        network.dynamics(perturbated,saver,max_iter)
        if (saver.history[-1]==patterns[index]).all() :
            retrieved_patterns += 1
    
    result_dict = {
        "network_size": size,
        "weights_rule": weight_rule,
        "num_patterns": num_patterns,
        "num_perturb": num_perturb,
        "match_frac": retrieved_patterns/num_trials
    }

    return result_dict

def experiment_robust(network, weight_rule, patterns, num_perturb, num_trials=100, max_iter=100):
    """
    Runs a robustness experiment for a given Hopfield network, weight rule,
    patterns, and number of perturbations.
    
    Parameters
    ----------
    network : HopfieldNetwork
        The Hopfield network to use in the experiment.
    weight_rule : str
        The weight rule to use for the network, either "hebbian" or "storkey".
    patterns : list[np.ndarray]
        The list of patterns to use in the experiment.
    num_perturb : float
        The fraction of neurons to perturb in each trial.
    num_trials : int
        The number of trials to run.
    max_iter : int
        The maximum number of iterations to run in each trial.
    
    Returns
    -------
    dict
        A dictionary containing the results of the experiment.
    """

    saver = DataSaver()
    num_patterns = len(patterns)
    retrieved_patterns = 0
    size = len(network.weights[0])

    for j in range(num_trials):
        index = random.choice(range(num_patterns))
        perturbated = perturb_pattern(patterns[index], floor(num_perturb*size))
        network.dynamics(perturbated,saver,max_iter)
        if (saver.history[-1]==patterns[index]).all() :
            retrieved_patterns += 1
    
    result_dict = {
        "network_size": size,
        "weights_rule": weight_rule,
        "num_patterns": num_patterns,
        "num_perturb": num_perturb,
        "match_frac": retrieved_patterns/num_trials
    }

    return result_dict


def capacity_experiment(weight_rule, fast):
    """
    Runs a capacity experiment for a given weight rule.
    
    Parameters
    ----------
    weight_rule : str
        The weight rule to use for the networks in the experiment, either "hebbian" or "storkey".
    
    Returns
    -------
    list: 
        A list of lists containing the results of the experiment for each network size.
    """
    if fast:
        sizes = [10, 18, 34, 63, 116, 215, 397]
    else:
        sizes = [10, 18, 34, 63, 116, 215, 397, 733, 1354, 2500]
    
    final_results = []
        
    for size in sizes:
        print(f"Beginning size {size}")
        t = num_patterns_generator(size,weight_rule)
        results = [experiment(size, num_patterns, weight_rule) for num_patterns in t]
        final_results.append(results)
    return final_results

def robustness_experiment(weight_rule, fast):
    """
    Runs a robustness experiment for a given weight rule.
    
    Parameters
    ----------
    weight_rule : str
        The weight rule to use for the networks in the experiment, either "hebbian" or "storkey".
    
    Returns
    -------
    list: 
        A list of lists containing the results of the experiment for each network size.
    """
    if fast:
        sizes = [10, 18, 34, 63, 116, 215, 397]
    else:
        sizes = [10, 18, 34, 63, 116, 215, 397, 733, 1354, 2500]

    final_results = []
    range_perturbs = np.arange(0.2,1,0.05)
    for size in sizes:
        results = []
        t = 2
        patterns = generate_patterns(t,size)
        network = HopfieldNetwork(patterns,weight_rule)
        for num_perturbs in range_perturbs:
            results.append(experiment_robust(network,weight_rule,patterns,num_perturbs))
        final_results.append(results)
    return final_results

def best_robustness(hebbian_results,storkey_results):
    """
    The function searches through the dictionaries in hebbian_results and storkey_results and returns two lists of dictionaries that contain the best results for each learning algorithm.
    A result is considered the best if it has a match_frac of at least 0.9 or a num_perturb of 0.2.
    The best result for each network size is the one that appears last in the respective list of dictionaries.

    Parameters
    ----------
    hebbian_results : list of lists of dictionaries
        List of lists of dictionaries containing the match fraction for hebbian networks of different sizes.
    storkey_results : list of lists of dictionaries
        List of lists of dictionaries containing the match fraction for storkey networks of different sizes.

    Returns
    -------
    list
        Two lists of dictionaries that contain the best results for Hebbian and Storkey
    """
    best_hebbian_results = []
    best_storkey_results = []
    for network_size in range(len(hebbian_results)):
        for trial in reversed(hebbian_results[network_size]):
            if trial["match_frac"] >= 0.9 or trial["num_perturb"] == 0.2:
                best_hebbian_results.append(trial)
                break
            
        for trial in reversed(storkey_results[network_size]):
            if trial["match_frac"] >= 0.9 or trial["num_perturb"] == 0.2:
                best_storkey_results.append(trial)
                break
    return best_hebbian_results,best_storkey_results

def plot_match_num(hebbian_results,storkey_results, out_path):
    """
    Plots the match fraction for hebbian and storkey networks with respect to the number of patterns for multiple network sizes and saves it.
    
    Parameters
    ----------
    hebbian_results : list of lists of dictionaries
        List of lists of dictionaries containing the match fraction for hebbian networks of different sizes.
    storkey_results : list of lists of dictionaries
        List of lists of dictionaries containing the match fraction for storkey networks of different sizes.
    out_path : str
        A string indicating the path to save the figure.
    """

    fig, axs = plt.subplots(2,5, sharey=True, figsize=(13,9))
    for network_size,ax in zip(range(len(hebbian_results)),axs.flat):
        hebbian_num_patterns = []
        hebbian_match_frac = []
        storkey_num_patterns = []
        storkey_match_frac = []
        for trial in range (len(hebbian_results[network_size])):
            hebbian_num_patterns.append(hebbian_results[network_size][trial]["num_patterns"])
            hebbian_match_frac.append(hebbian_results[network_size][trial]["match_frac"])
            storkey_num_patterns.append(storkey_results[network_size][trial]["num_patterns"])
            storkey_match_frac.append(storkey_results[network_size][trial]["match_frac"])
        ax.set_title("Network of size : "+str(hebbian_results[network_size][0]["network_size"]))
        ax.plot(hebbian_num_patterns, hebbian_match_frac, label = "hebbian")
        ax.plot(storkey_num_patterns, storkey_match_frac, label = "storkey")
        ax.legend()
    axs[0,0].set_ylabel('Match fraction')
    axs[1,0].set_ylabel('Match fraction')
    for i in range(5):
        axs[1,i].set_xlabel("Number of paterns")
    fig.suptitle("Capacity test")
    # Checks if the path contains a folder, and create it if it doesn't exist yet
    folder = out_path.split('/')[0]
    if folder != out_path:
        if not os.path.isdir(folder):
            os.mkdir(folder)
    plt.savefig(out_path)

def plot_match_perturb(hebbian_results, storkey_results, out_path):
    """
    Plots the match fraction for hebbian and storkey networks with respect to the percentage of pertubation for multiple network sizes and saves it.
    
    Parameters
    ----------
    hebbian_results : list of lists of dictionaries
        List of lists of dictionaries containing the match fraction for hebbian networks of different sizes.
    storkey_results : list of lists of dictionaries
        List of lists of dictionaries containing the match fraction for storkey networks of different sizes.
    out_path : str
        A string indicating the path to save the figure.
    """

    # Create a pandas DataFrame from the results dictionaries
    hebbian_df = pd.DataFrame(hebbian_results)
    storkey_df = pd.DataFrame(hebbian_results)
    # Save dataframe as an hdf5 file
    hebbian_df.to_hdf(out_path.split("/")[0]+"/hebbian_perturb_results.hdf", key='df')
    storkey_df.to_hdf(out_path.split("/")[0]+"/storkey_perturb_results.hdf", key='df')

    fig, axs = plt.subplots(2,5, sharey=True, figsize=(13,9))
    for network_size,ax in zip(range(len(hebbian_results)),axs.flat):
        hebbian_num_perturb = []
        hebbian_match_frac = []
        storkey_num_perturb = []
        storkey_match_frac = []
        for trial in range (len(hebbian_results[network_size])):
            hebbian_num_perturb.append(hebbian_results[network_size][trial]["num_perturb"])
            hebbian_match_frac.append(hebbian_results[network_size][trial]["match_frac"])
            storkey_num_perturb.append(storkey_results[network_size][trial]["num_perturb"])
            storkey_match_frac.append(storkey_results[network_size][trial]["match_frac"])
        ax.set_title("Network of size : "+str(hebbian_results[network_size][0]["network_size"]))
        ax.plot(hebbian_num_perturb, hebbian_match_frac, label = "hebbian")
        ax.plot(storkey_num_perturb, storkey_match_frac, label = "storkey")
        ax.legend()
    axs[0,0].set_ylabel('Match fraction')
    axs[1,0].set_ylabel('Match fraction')
    for i in range(5):
        axs[1,i].set_xlabel('Percentage of perturbation')
    fig.suptitle("Robustness test")
    # Checks if the path contains a folder, and create it if it doesn't exist yet
    folder = out_path.split('/')[0]
    if folder != out_path:
        if not os.path.isdir(folder):
            os.mkdir(folder)
    plt.savefig(out_path)

def plot_capacity(hebbian_results,storkey_results, out_path):
    """
    Plots the capacity for hebbian and storkey networks with respect to the network sizes and saves it.
    
    Parameters
    ----------
    hebbian_results : list of lists of dictionaries
        List of lists of dictionaries containing the match fraction for hebbian networks of different sizes.
    storkey_results : list of lists of dictionaries
        List of lists of dictionaries containing the match fraction for storkey networks of different sizes.
    out_path : str
        A string indicating the path to save the figure.
    """
    hebbian_capacities = []
    storkey_capacities = []
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    network_sizes =[]
    for network_size in range(len(hebbian_results)):
        for trial in range (len(hebbian_results[network_size])):
            if hebbian_results[network_size][trial]["match_frac"]>=0.9:
                hebbian_num_patterns =hebbian_results[network_size][trial]["num_patterns"]
                break
        for trial in range (len(hebbian_results[network_size])):
            if storkey_results[network_size][trial]["match_frac"]>=0.9:
                storkey_num_patterns = storkey_results[network_size][trial]["num_patterns"]
                break
            storkey_num_patterns = 0
        hebbian_capacities.append(hebbian_num_patterns)
        storkey_capacities.append(storkey_num_patterns)
        network_sizes.append(hebbian_results[network_size][0]["network_size"])

    hebbian_expected_capacities = [x/(2*log(x)) for x in network_sizes]
    storkey_expected_capacities = [x/sqrt((2*log(x))) for x in network_sizes]

    ax.plot(network_sizes,hebbian_capacities, label = "hebbian")
    ax.plot(network_sizes, storkey_capacities, label = "storkey")
    ax.plot(network_sizes,hebbian_expected_capacities, label = "hebbian expectation")
    ax.plot(network_sizes,storkey_expected_capacities, label = "storkey expectation")
    ax.legend()
    # Checks if the path contains a folder, and create it if it doesn't exist yet
    folder = out_path.split('/')[0]
    if folder != out_path:
        if not os.path.isdir(folder):
            os.mkdir(folder)
    plt.savefig(out_path)
    

if __name__ == "__main__":
    import doctest # Importing the library
    print("Starting doctests") # not required (just for clarity in output)
    doctest.testmod() 