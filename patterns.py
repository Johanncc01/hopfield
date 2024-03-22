import numpy as np
from math import *
from PIL import Image
                                        # Patterns Function

def generate_patterns(num_patterns, pattern_size):
    """
    Generates random binary patterns with values in {-1,1} of specific size.

    Parameters
    ----------
    num_patterns : int
        Number of patterns to create
    pattern_size : int
        Size of one binary pattern

    Returns
    -------
    Numpy array (2D)
        Array (of shape : (num_patterns, pattern_size)) in which the rows are the generated patterns of desired size
        
    Raises
    ------
    ValueError
        If the number of patterns or the pattern size are not positive integers.

    Examples
    -------
    >>> len(generate_patterns(5,8))
    5
    >>> len(generate_patterns(5,8)[0])
    8
    """

    # Exceptions for the integers parameters (positive integers)
    if num_patterns <= 0 or pattern_size <= 0:
        raise ValueError("Both number and size values have to be positive.")
    if not(float(num_patterns).is_integer()) and not(float(pattern_size).is_integer()):
        raise ValueError("Both number and size values must be integers.")

    return np.random.choice((-1.,1.),num_patterns*pattern_size).reshape(num_patterns,pattern_size)

def perturb_pattern(pattern, num_perturb):
    """
    Samples a given number of elements of the imput binary patter at random and changes their sign.

    Parameters
    ----------
    pattern : numpy array (1D)
        Binary pattern to perturbate
    num_perturb : int
        Number of modified elements

    Returns
    -------
    Numpy array (1D)
        Copy of the pattern with disturbed elements

    Raises
    ------
    ValueError
        If the pattern is empty or not binary.
    ValueError
        If the number of perturbations is not a positive integers.
    ValueError
        If the number of perturbations is larger than the size of the pattern.
    """
    # Exceptions for the pattern (not empty and binary)
    if pattern.size == 0:
        raise ValueError("The pattern must not be empty.")
    if ((pattern!=-1) & (pattern!=1)).any():
        raise ValueError("The pattern must be binary, i.e. -1 or 1.")

    # Exceptions for the perturbations number (positive integer)
    if num_perturb < 0:
        raise ValueError("The number of perturbations needs to be positive.")
    if not(float(num_perturb).is_integer()):
        raise ValueError("The number of perturbations must be an integer.")
    if num_perturb > len(pattern):
        raise ValueError("The number of perturbations cannot be larger than the size of the pattern.")
    
    
    new_pattern = np.copy(pattern)
    arr = np.ones(len(new_pattern))
    arr[:num_perturb] = -1.
    np.random.shuffle(arr)
    return new_pattern*arr

def pattern_match(memorized_patterns, pattern):
    """
    Retrieves the index of a given pattern in a memorized list of patterns, if a match is found.

    Parameters
    ----------
    memorized_patterns : numpy array (2D)
        Memorized patterns to analyse
    pattern : numpy array (1D)
        Pattern to match 

    Returns
    -------
    int
        Index of the line corresponding to the pattern, if the matching pattern has been found. Otherwise, returns None.

    Raises
    ------
    ValueError
        If the pattern to match is empty or not binary.
    ValueError
        If the memorized patterns are empty.
    ValueError
        If the size of the pattern to match and the memorized patterns are not the same.
    """
    # Exceptions (non-zero pattern and same pattern_size)
    if pattern.size == 0:
        raise ValueError("The pattern must not be empty.")
    if ((pattern!=-1) & (pattern!=1)).any():
        raise ValueError("The pattern must be binary, i.e. -1 or 1.")
    if memorized_patterns.size == 0:
        raise ValueError("The memorized patterns must not be empty.")
    if np.shape(memorized_patterns)[1] != len(pattern):
        raise ValueError("The pattern size should be the same in the given pattern and in the memorized patterns.")

    for i in range(len(memorized_patterns)):
        if (np.equal(pattern, memorized_patterns[i]).all()):
            return i
                                    
def generate_checkboard(tile_size, grid_size):
    """
    Create a checkboard where white tiles are represented with 1, and black tiles with -1.

    Parameters
    ----------
    tile_size : int
        Size of a single tile which constitute the checkboard
    grid_size : int
        Size of the total grid, i.e. the number of tiles of the checkboard

    Returns
    -------
    numpy array (2D)
        Full generated checkboard (which is a bianry array)

    Raises
    ------
    ValueError
        If the sizes are not positive integers
    ValueError
        If the tile_size is greater than the grid_size
    """
    # Exeptions for the sizes of the checkboard (must be positive integers)
    if tile_size <= 0 or grid_size <= 0:
        raise ValueError("Both size values have to be positive.")
    if not(float(tile_size).is_integer()) and not(float(grid_size).is_integer()):
        raise ValueError("Both size values must be integers.")

    # Exeptions if tile_size is higher than grid_size
    if tile_size > grid_size:
        raise ValueError("The tile_size cannot be greater than the grid_size.")

    # Create a white tile filled with ones
    white_tile = np.ones((tile_size, tile_size))
    # Create a black tile filled with negative ones
    black_tile = np.negative(white_tile)
    
    rows = []
    for i in range(grid_size):
        row = []
        for j in range(grid_size):
            if (i + j) % 2 == 0:
                row.append(white_tile)
            else:
                row.append(black_tile)
        rows.append(np.concatenate(row, axis=1))

    checkboard = np.concatenate(rows, axis=0)

    return checkboard

    
import numpy as np

def image_to_pattern(path):
    """
    Converts an image to a binary pattern, based on the colors of the pixels.

    Parameters
    ----------
    path : string
        Path to the image to read

    Returns
    -------
    numpy array (1D)
        Flat inary pattern based on the pixels of the image

    Raises
    ------
    ValueError
        If the image format is unsupported by PIL
    """
    try:
        im_gray = np.array(Image.open(path).convert("L"))
    except Image.UnidentifiedImageError:
        raise ValueError('The image format is unsupported by PIL. Try again with a .jpg or .png image.')

    return np.where(im_gray < 126, -1., 1. )


if __name__ == "__main__":  # pragma: no cover
    import doctest # Importing the library
    print("Starting doctests") # not required (just for clarity in output)
    doctest.testmod() 
