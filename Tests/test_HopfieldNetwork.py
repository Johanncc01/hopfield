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
import numpy as np



def test_init():
    # Checks if ValueError is raised when init with empty patterns
    with pytest.raises(ValueError, match="The patterns must not be empty."):
        HopfieldNetwork(np.array([[]]))

    # Checks that the rule is Hebbian or Storkey
    with pytest.raises(ValueError, match="The rule should be either hebbian or storkey."):
        HopfieldNetwork(np.ones((1,3)), "unknown")

def test_hebbian(benchmark):
    num_pattern = random.randint(3, 100)
    pattern_size = random.randint(10, 1000)
    patterns = generate_patterns(num_pattern, pattern_size)
    network = HopfieldNetwork(patterns, "hebbian")

    # Benchmark 
    benchmark.pedantic(HopfieldNetwork.hebbian_weights, args=(network, patterns), rounds=5, iterations=1)

    # Checks if the Hebbian weights have the correct size
    shape = np.shape(network.weights)
    assert (shape[0] == pattern_size)
    assert (shape[1] == pattern_size)

    # Checks that the values are in the correct range, i.e. [-1,1]
    assert (np.amin(network.weights)>=-1 and np.amax(network.weights)<=1)

    # Checks if the matrix is symetric
    assert (network.weights==network.weights.T).all()

    # Checks that the diagonal is empty
    assert not np.diag(network.weights).any()

    # Computation exemple
    expected = np.array([[0., 0.33333333, -0.33333333, -0.33333333],
                        [0.33333333, 0., -1, 0.33333333],
                        [-0.33333333, -1, 0., -0.33333333],
                        [-0.33333333, 0.33333333, -0.33333333, 0.]])
    test = HopfieldNetwork(np.array([[1, 1, -1, -1],[1, 1, -1, 1],[-1, 1, -1, 1]]), "hebbian")
    computed = test.weights
    assert np.allclose(computed, expected)

    # Checks if ValueError is raised when computing the weights of empty patterns
    with pytest.raises(ValueError, match="The patterns must not be empty."):
        HopfieldNetwork(np.array([[]]), "hebbian")

def test_storkey(benchmark):
    num_pattern = random.randint(3, 100)
    pattern_size = random.randint(10, 1000)
    patterns = generate_patterns(num_pattern, pattern_size)
    network = HopfieldNetwork(patterns, "hebbian")

    # Benchmark 
    benchmark.pedantic(HopfieldNetwork.storkey_weights, args=(network, patterns), rounds=5, iterations=1)

    # Checks if the Strokey weights have the correct size
    shape = np.shape(network.weights)
    assert (shape[0] == pattern_size)
    assert (shape[1] == pattern_size)

    # Computation exemple
    expected = np.array([[1.125, 0.25, -0.25, -0.5],
                        [0.25, 0.625, -1, 0.25],
                        [-0.25, -1, 0.625, -0.25],
                        [-0.5, 0.25, -0.25, 1.125]])
    test = HopfieldNetwork(np.array([[1, 1, -1, -1],[1, 1, -1, 1],[-1, 1, -1, 1]]), "storkey")
    computed = test.weights
    assert np.allclose(computed, expected)

    # Checks if ValueError is raised when computing the weights of empty patterns
    with pytest.raises(ValueError, match="The patterns must not be empty."):
        HopfieldNetwork(np.array([[]]), "storkey")

def test_update(benchmark):
    num_pattern = random.randint(3, 100)
    pattern_size = random.randint(10, 1000)
    patterns = generate_patterns(num_pattern, pattern_size)
    network = HopfieldNetwork(patterns)

    nb_changes = random.randint(1,len(patterns[0]))
    perturbated = perturb_pattern(patterns[0], nb_changes)
    
    sync = network.update(perturbated)

    # Benchmarks
    benchmark.pedantic(network.update, args=(perturbated,), iterations=100)

    # Check that the returned arrays still have the same shape
    assert np.shape(perturbated) == np.shape(sync)

    # Check that computed update is still a binary array
    assert ((sync==-1) | (sync==1)).all()

    # Checks if ValueError is raised when updating an empty state or with an empty matrix
    with pytest.raises(ValueError, match="The state or the weights must not be empty"):
        network.update(np.array([]))

def test_update_async(benchmark):
    num_pattern = random.randint(3, 100)
    pattern_size = random.randint(10, 1000)
    patterns = generate_patterns(num_pattern, pattern_size)
    network = HopfieldNetwork(patterns)

    nb_changes = random.randint(1,len(patterns[0]))
    perturbated = perturb_pattern(patterns[0], nb_changes)
    
    a_sync = network.update_async(perturbated)

    # Benchmark
    benchmark.pedantic(network.update_async, args=(perturbated,), iterations=100)

    # Check that the returned arrays still have the same shape
    assert np.shape(perturbated) == np.shape(a_sync)

    # Check that computed update is still a binary array
    assert ((a_sync==-1) | (a_sync==1)).all()

    # Checks if ValueError is raised when updating an empty state or with an empty matrix
    with pytest.raises(ValueError, match="The state or the weights must not be empty"):
        network.update_async(np.array([]))

def test_dynamics():
    num_pattern=50
    pattern_size=2500
    nb_changes=1000
    max_iter_sync=20
    max_iter_async=30000
    conv_iter=10000
    skip=1000
    
    patterns = generate_patterns(num_pattern, pattern_size)
    hebbian_network = HopfieldNetwork(patterns, "hebbian")
    storkey_network = HopfieldNetwork(patterns, "storkey")

    index = random.randint(0,num_pattern-1)
    perturbated = perturb_pattern(patterns[index], nb_changes)

    hebbian_sync_data = DataSaver()
    hebbian_async_data = DataSaver()
    storkey_sync_data = DataSaver()
    storkey_async_data = DataSaver()

    hebbian_network.dynamics(perturbated,hebbian_sync_data,max_iter_sync)
    storkey_network.dynamics(perturbated,storkey_sync_data,max_iter_sync)
    hebbian_network.dynamics_async(perturbated,hebbian_async_data,max_iter_async,conv_iter,skip)
    storkey_network.dynamics_async(perturbated,storkey_async_data,max_iter_async,conv_iter,skip)

    # Checking if the number of iteration before converge is smaller than the max iterations
    assert len(hebbian_sync_data.history)-1 <= max_iter_sync
    assert len(storkey_sync_data.history)-1 <= max_iter_sync

    assert len(hebbian_async_data.history)-1 <= max_iter_async
    assert len(storkey_async_data.history)-1 <= max_iter_async

    # Checking if the pattern has converged towards the memorized one
    assert np.allclose(hebbian_sync_data.history[-1], patterns[index])
    assert np.allclose(storkey_sync_data.history[-1], patterns[index])

    assert np.allclose(hebbian_async_data.history[-1], patterns[index])
    assert np.allclose(storkey_async_data.history[-1], patterns[index])

    # Checking that the convergance pattern matches with the right index in memorized patterns
    assert pattern_match(patterns, hebbian_sync_data.history[-1]) == index
    assert pattern_match(patterns, storkey_sync_data.history[-1]) == index

    assert pattern_match(patterns, hebbian_async_data.history[-1]) == index
    assert pattern_match(patterns, storkey_async_data.history[-1]) == index

    # Checking if ValueError is raised when empty/non binary states
    with pytest.raises(ValueError, match="The initial state must not be empty."):
        hebbian_network.dynamics(np.array([]),hebbian_sync_data,max_iter_sync)
    with pytest.raises(ValueError, match="The initial state must not be empty."):
        hebbian_network.dynamics_async(np.array([]),hebbian_sync_data,max_iter_async,conv_iter,skip)

    with pytest.raises(ValueError, match="The pattern must be binary, i.e. -1 or 1."):
        hebbian_network.dynamics(np.array([1,2,3]),hebbian_sync_data,max_iter_sync)
    with pytest.raises(ValueError, match="The pattern must be binary, i.e. -1 or 1."):
        hebbian_network.dynamics_async(np.array([1,2,3]),hebbian_sync_data,max_iter_async,conv_iter,skip)

    # Checking if ValueError is raised when using non positive integers for maximal number of iterations and skip
    with pytest.raises(ValueError, match="The maximal number of iterations has to be positive."):
        hebbian_network.dynamics(perturbated,hebbian_sync_data,-45)
    with pytest.raises(ValueError, match="Maximal iterations values have to be positive."):
        hebbian_network.dynamics_async(perturbated,hebbian_sync_data,-23,-9300,-3)
    
    with pytest.raises(ValueError, match="The number of perturbations must be an integer."):
        hebbian_network.dynamics(perturbated,hebbian_sync_data,0.5)
    with pytest.raises(ValueError, match="Maximal iterations values must be integers."):
        hebbian_network.dynamics_async(perturbated,hebbian_sync_data,23.4,56.4,1.1)

    # Checking if ValueError is raised when conv_iter is larger than max_iter
    with pytest.raises(ValueError, match="Convergence max value must be lower than global max value."):
        hebbian_network.dynamics_async(perturbated,hebbian_sync_data,300,500,skip)


if __name__ == "__main__":  # pragma: no cover
    pytest.main()