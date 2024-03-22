# setting path to import from parent folder
import sys, os, inspect
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from patterns import *
import pytest
import random
import numpy as np



def test_generation():
    num_pattern = random.randint(3, 100)
    pattern_size = random.randint(10, 1000)
    patterns = generate_patterns(num_pattern, pattern_size)

    # Checks if the matrix has good shape for random values
    shape = np.shape(patterns)
    assert (shape[0] == num_pattern)
    assert (shape[1] == pattern_size)

    # Checks if the patterns only contains -1 or 1
    assert ((patterns==-1) | (patterns==1)).all()

    # Checks if ValueError is raised when passing negative or decimal values
    with pytest.raises(ValueError, match="Both number and size values have to be positive."):
        generate_patterns(-1,-1)
    with pytest.raises(ValueError, match="Both number and size values must be integers."):
        generate_patterns(2.3,5.2)
    
def test_perturbation():
    num_pattern = random.randint(3, 100)
    pattern_size = random.randint(10, 1000)
    patterns = generate_patterns(num_pattern, pattern_size)
    base_pattern = patterns[random.randint(0,len(patterns)-1)]
    nb_changes = random.randint(1,len(patterns[0]))
    perturbated = perturb_pattern(base_pattern, nb_changes)

    # Checks if the perturbated pattern only contains -1 or 1
    assert ((perturbated==-1) | (perturbated==1)).all()

    # Checks if the pattern was correctly perturbated
    diff = base_pattern - perturbated
    assert np.count_nonzero(diff) == nb_changes
    assert (base_pattern!=perturbated).any()

    # Checks if 0 pertrubations returns the base pattern
    assert np.allclose(base_pattern, perturb_pattern(base_pattern, 0))

    # Checks if ValueError is raised when the pattern is empty or non-binary
    with pytest.raises(ValueError, match="The pattern must not be empty."):
        perturb_pattern(np.array([]), 0)
    with pytest.raises(ValueError, match="The pattern must be binary, i.e. -1 or 1."):
        perturb_pattern(np.array([2,3]), 0)

    # Checks if ValueError is raised when the nb_changes is not a positive integer
    with pytest.raises(ValueError, match="The number of perturbations needs to be positive."):
        perturb_pattern(base_pattern, -3)
    with pytest.raises(ValueError, match="The number of perturbations must be an integer."):
        perturb_pattern(base_pattern, 2.5)
    with pytest.raises(ValueError, match="The number of perturbations cannot be larger than the size of the pattern."):
        perturb_pattern(np.array([1,1,1]), 4)    

def test_match():
    num_pattern = random.randint(3, 100)
    pattern_size = random.randint(10, 1000)
    patterns = generate_patterns(num_pattern, pattern_size)
    
    # Checks if a memorized pattern has a match with the correct index
    id = random.randint(0, len(patterns)-2)
    assert pattern_match(patterns, patterns[id]) == id

    # Checks if a different pattern has a match
    assert pattern_match(np.ones((num_pattern, pattern_size)), np.full((pattern_size), -1)) is None

    # Checks that if there are multiples matches, the returned row index is the first one.
    target = np.random.choice((-1.,1), pattern_size)
    ones = np.ones((num_pattern, pattern_size))
    ones[id] = target
    ones[id+1] = target
    assert pattern_match(ones, target) == id

    # Checks if ValueError is raised when the pattern is empty, non-binary or has not the right size
    with pytest.raises(ValueError, match="The pattern must not be empty."):
        pattern_match(patterns, np.array([]))
    with pytest.raises(ValueError, match="The pattern must be binary, i.e. -1 or 1."):
        pattern_match(np.array([1,1,1]), np.array([1,2,3]))
    with pytest.raises(ValueError, match="The memorized patterns must not be empty."):
        pattern_match(np.array([]), patterns[0])
    with pytest.raises(ValueError, match="The pattern size should be the same in the given pattern and in the memorized patterns."):
        pattern_match(np.array([[1,1,1],[1,1,1]]), np.array([1,-1,1,1,-1]))
    

if __name__ == "__main__":  # pragma: no cover
    pytest.main()