# setting path to import from parent folder
import sys, os, inspect
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from patterns import *
from HopfieldNetwork import *
from DataSaver import *
import pytest




def test_energy(benchmark):
    num_pattern = 50
    pattern_size = 2500
    nb_changes = 1000
    max_iter_async = 30000
    conv_iter = 10000
    skip = 1000
    path = "plot.png"

    patterns = generate_patterns(num_pattern, pattern_size)
    network = HopfieldNetwork(patterns)
    perturbated = perturb_pattern(patterns[0], nb_changes)

    data = DataSaver()
    network.dynamics_async(perturbated, data, max_iter_async, conv_iter, skip)

    # Benchmark
    benchmark.pedantic(data.compute_energy, args=(perturbated, network.weights,), iterations=5)

    # Checks if the energy values are decreasing with number of iterations
    assert all(earlier >= later for earlier, later in zip(data.energy_values, data.energy_values[1:]))

    # Checks if ValueError is raised when the given state is empty or non-binary
    with pytest.raises(ValueError, match="The state must not be empty."):
        data.compute_energy(np.array([]), network.weights)
    with pytest.raises(ValueError, match="The pattern must be binary, i.e. -1 or 1."):
        data.compute_energy(np.array([1,2,3,-2]), network.weights)

    # Checks if a file has been generated and clear it
    data.plot_energy(path)
    assert os.path.isfile(path)
    os.remove(path)

    # Check if ValueError is raised when the given list is empty
    with pytest.raises(ValueError, match="The energy values list must not be empty."):
        empty = DataSaver()
        empty.plot_energy(path)

    # Check if ValueError is raised with an incorrect file type (not .jpeg or .png)
    with pytest.raises(ValueError, match="The saving path must end with .jpeg or .png"):
        data.plot_energy("plot.txt")


if __name__ == "__main__":
    pytest.main()