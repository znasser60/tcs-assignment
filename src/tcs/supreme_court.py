import numpy as np 

class SupremeCourt: 
    def __init__(self, file_path): 
        """
        Initialize a SupremeCourt object.
        """
        self.file_path = file_path 

    def load_data(self): 
        """
        Load data from the file path.
        """
        self.data = np.loadtxt(self.file_path)
    
    