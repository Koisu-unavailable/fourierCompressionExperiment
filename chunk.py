import numpy as np
import utils

def chunk(data: np.ndarray, chunk_size: int):
    """
    Splits the data into chunks of the specified size.
    
    Args:
        data (np.ndarray): The input data to be chunked.
        chunk_size (int): The size of each chunk.
        
    Returns:
        list: A list of chunks, where each chunk is a numpy array.
    """
    if utils.is_mono(data):
        return [data[i:(i + chunk_size)] for i in range(0, len(data), chunk_size)]
    else:
        return [data[:, 0][i:i + chunk_size] for i in range(0, len(data), chunk_size)]