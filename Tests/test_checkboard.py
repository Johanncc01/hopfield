# setting path to import from parent folder
import sys, os, inspect
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from patterns import *
from HopfieldNetwork import *
from DataSaver import *
import pytest
import random

import matplotlib.pyplot as plt

def test_checkboard_generation():
    tile_size = random.randint(1,5)
    grid_size = random.randint(6,20)
    checkboard = generate_checkboard(tile_size, grid_size)
    showFig = False

    # Checks if the generated checkboard is a binary pattern (-1/1)
    assert ((checkboard==-1) | (checkboard==1)).all()

    # Checks if the generated checkboard has the correct size
    assert (len(checkboard) == tile_size*grid_size) 
    assert (len(checkboard[0]) == tile_size*grid_size)

    # Checks if ValueError when passing negative or decimal values
    with pytest.raises(ValueError, match="Both size values have to be positive."):
        generate_checkboard(-1,-1)
    with pytest.raises(ValueError, match="Both size values must be integers."):
        generate_checkboard(2.3,5.2)

    # Checks if ValueError when passing negative or decimal values
    with pytest.raises(ValueError, match="The tile_size cannot be greater than the grid_size."):
        generate_checkboard(10, 1)

    # Shows the generated checkboard if showFig is set to True
    fig, ax = plt.subplots()
    ax.imshow(checkboard, cmap='gray')
    ax.set_axis=False
    if showFig:
        plt.show()  # pragma: no cover

def test_checkboard_export():
    num_pattern = 2500
    pattern_size = 50
    nb_changes = 1000
    max_iter_sync = 20
    max_iter_async = 30000
    conv_iter = 10000
    skip = 1000
    path_to_output = "outputs/"

    patterns = generate_patterns(pattern_size, num_pattern)
    patterns[0] = generate_checkboard(5,10).reshape(1, 2500)

    network = HopfieldNetwork(patterns, "hebbian")

    perturbated = perturb_pattern(patterns[0], nb_changes)

    # Generation of sync and async state list
    sync_data = DataSaver()
    async_data = DataSaver()

    network.dynamics(perturbated, sync_data, max_iter_sync)
    network.dynamics_async(perturbated, async_data, max_iter_async, conv_iter, skip)

    sync_data.save_video(path_to_output+"hebbian_sync.gif")
    async_data.save_video(path_to_output+"hebbian_async.gif") 

    # Checks if a file has been generated
    assert os.path.isfile(path_to_output+"hebbian_sync.gif")
    assert os.path.isfile(path_to_output+"hebbian_async.gif")

    # Check if ValueError is raised when the given list is empty
    with pytest.raises(ValueError, match="The state list must not be empty."):
        empty = DataSaver()
        empty.save_video("path.gif")

    # Check if ValueError is raised with an incorrect file type (not .gif or .mp4)
    with pytest.raises(ValueError, match="The saving path must end with .gif or .mp4."):
        sync_data.save_video("path.txt")


if __name__ == "__main__":  # pragma: no cover
    pytest.main()