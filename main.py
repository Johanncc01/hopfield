from patterns import *
from Tests.test_checkboard import *
from HopfieldNetwork import *
from DataSaver import *
from experiment import *

def checkboard():
    """
    Runs a checkboard recalling experiment using Hopfield networks.
    
    Example
    -------
        >>> checkboard()
        # This runs the checkboard recalling experiment and generates several
        # JPEG files in the "Outputs" folder.
    """
    num_pattern = 2500
    pattern_size = 50
    nb_changes = 1000
    max_iter_sync = 20
    max_iter_async = 30000
    conv_iter = 10000
    skip = 1000
    path_to_output = "Outputs/"

    patterns = generate_patterns(pattern_size, num_pattern)
    patterns[0] = generate_checkboard(5, 10).reshape(1, 2500)

    hebbian_network = HopfieldNetwork(patterns,"hebbian")
    storkey_network = HopfieldNetwork(patterns,"storkey")
    
    hebbian_sync_data = DataSaver()
    hebbian_async_data = DataSaver()
    storkey_sync_data = DataSaver()
    storkey_async_data = DataSaver()

    perturbated = perturb_pattern(patterns[0], nb_changes)

    # Generation of sync state list
    hebbian_network.dynamics(perturbated,hebbian_sync_data,max_iter_sync)
    storkey_network.dynamics(perturbated,storkey_sync_data,max_iter_sync)

    # Generation of async state list
    hebbian_network.dynamics_async(perturbated,hebbian_async_data,max_iter_async,conv_iter,skip)
    storkey_network.dynamics_async(perturbated,storkey_async_data,max_iter_async,conv_iter,skip)

    # Save videos
    hebbian_sync_data.save_video(path_to_output+"hebbian_sync.gif")
    hebbian_async_data.save_video(path_to_output+"hebbian_async.gif") 
    storkey_sync_data.save_video(path_to_output+"storkey_sync.gif") 
    storkey_async_data.save_video(path_to_output+"storkey_async.gif")

    #plot energy
    hebbian_sync_data.plot_energy(path_to_output+"hebbian_sync.jpeg",'Hebbian sync energy')
    hebbian_async_data.plot_energy(path_to_output+"hebbian_async.jpeg",'Hebbian async energy')
    storkey_sync_data.plot_energy(path_to_output+"storkey_sync.jpeg",'Storkey sync energy')
    storkey_async_data.plot_energy(path_to_output+"storkey_async.jpeg",'Storkey async energy')

def image_recalling(input_path, square = False):
    """
    Runs an image recalling experiment using a Hopfield network.
    
    Parameters
    ----------
    path : string
        The path of the image to perturb. The image should be in the "Inputs" folder.
        The path should be relative to the imput folder, i.e. if the image is at the root of the folder, the argument should be only the name and extension of the file.
        The image format must be supported by PIL.image (.jpg, .png, ...)
    square : bool, optionnal
        Whether to retrieve from a square part of the image or from the image randomly perturbed. Default is False.
    
    Example
    -------
        >>> image_recalling(image.jpg)
        # This runs the image recalling experiment using a rectangular image and
        # generates several GIF files in the "Outputs" folder.
        
        >>> image_recalling(image.jpg,square=True)
        # This runs the image recalling experiment using a square image and
        # generates several GIF files in the "Outputs" folder.
    """
    num_pattern = 10000
    pattern_size = 50
    nb_changes = 3000
    max_iter_sync = 50
    max_iter_async = 80000
    conv_iter = 30000
    skip = 1000
    path_to_input = "Inputs/"
    path_to_output = "Images/"
    name = ""

    image = image_to_pattern(path_to_input+input_path)
    patterns = generate_patterns(pattern_size, num_pattern)
    
    patterns[0] = image.reshape(1,10000)

    hebbian_network = HopfieldNetwork(patterns,"hebbian")
    storkey_network = HopfieldNetwork(patterns,"storkey")
    
    hebbian_sync_data = DataSaver()
    hebbian_async_data = DataSaver()
    storkey_sync_data = DataSaver()
    storkey_async_data = DataSaver()

    if square : 
        perturbated = patterns[0].reshape(100,100)
        perturbated = np.block([[perturbated[:50,:50], np.full((50,50),-1)],[np.full((50,100),-1)]]).reshape(1,10000)
        perturbated = perturbated.reshape(10000,1)
        name = "square_"+input_path.split(".")[0]+".gif"
    else :
        perturbated = perturb_pattern(patterns[0], nb_changes)
        name = input_path.split(".")[0]+".gif"
        
    # Generation of sync state list
    hebbian_network.dynamics(perturbated,hebbian_sync_data,max_iter_sync)
    storkey_network.dynamics(perturbated,storkey_sync_data,max_iter_sync)

    # Generation of async state list
    hebbian_network.dynamics_async(perturbated,hebbian_async_data,max_iter_async,conv_iter,skip)
    storkey_network.dynamics_async(perturbated,storkey_async_data,max_iter_async,conv_iter,skip)

    # Save videos
    hebbian_sync_data.save_video(path_to_output+"hebbian_sync_"+name,(100,100))
    hebbian_async_data.save_video(path_to_output+"hebbian_async_"+name,(100,100))
    storkey_sync_data.save_video(path_to_output+"storkey_sync_"+name,(100,100))
    storkey_async_data.save_video(path_to_output+"storkey_async_"+name,(100,100))

def main_experiment(output = "Results/", fast = True):
    """
    Runs a capacity experiment using Hopfield networks and generates several
    plots and files.
    
    Parameters
    ----------
    output : str, optionnal
        The directory where the output files should be saved. Default is "Results/"
    fast : True, optionnal
        If True, runs the experiments on a smaller range of sizes to reduce computation times. Default is True.
    Returns
    -------
        None
    
    Example
    -------
        >>> main_experiment()
        # This runs the capacity experiment using the default output directory
        # and generates several files and plots in that directory.
        
        >>> main_experiment(output="my_results/")
        # This runs the capacity experiment using the "my_results" directory
        # as the output directory and generates several files and plots in that
        # directory.
    """

    capacity_hebbian_results = capacity_experiment("hebbian", fast)
    capacity_storkey_results = capacity_experiment("storkey", fast)

    robustness_hebbian_results = robustness_experiment("hebbian", fast)
    robustness_storkey_results = robustness_experiment("storkey", fast)

    plot_match_perturb(robustness_hebbian_results,robustness_storkey_results,output+"robustness.jpeg")
    
    robustness_hebbian_results,robustness_storkey_results = best_robustness(robustness_hebbian_results,robustness_storkey_results)
    
    # Create a pandas DataFrame from the results dictionaries
    capacity_hebbian_df = pd.DataFrame(capacity_hebbian_results)
    capacity_storkey_df = pd.DataFrame(capacity_storkey_results)
    robustness_hebbian_df = pd.DataFrame(robustness_hebbian_results)
    robustness_storkey_df = pd.DataFrame(robustness_storkey_results)

    # Save dataframe as an hdf5 file
    capacity_hebbian_df.to_hdf(output+"hebbian_capacity_results.hdf", key='df')
    capacity_storkey_df.to_hdf(output+"storkey_capacity_results.hdf", key='df')
    robustness_hebbian_df.to_hdf(output+"hebbian_robustness_results.hdf", key='df')
    robustness_storkey_df.to_hdf(output+"storkey_robustness_results.hdf", key='df')

    # Save dataframe as an md file
    robustness_hebbian_df.to_markdown(output+"hebbian_robustness_results.md")
    robustness_storkey_df.to_markdown(output+"storkey_robustness_results.md")

    # Print dataframe in md format
    print(robustness_hebbian_df.to_markdown())
    print(robustness_storkey_df.to_markdown())

    plot_capacity(capacity_hebbian_results,capacity_storkey_results, output+"capacities.jpeg")
    plot_match_num(capacity_hebbian_results,capacity_storkey_results, output+"matches.jpeg")
    

# checkboard()
# image_recalling("cervin.png", True)
# image_recalling("animal.jpeg", False)
main_experiment()
