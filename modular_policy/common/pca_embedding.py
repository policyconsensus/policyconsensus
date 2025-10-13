import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class PCAEmbedding:
    def __init__(self,
                 n_components=15,
                 normalize = True,
                 mode = 'Trian',
                 store = False,
                 transformation_matrix_path = None,  # Path to the PCA transformation matrix
                 mean_matrix_path = None
                 ):

        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.normalize = normalize
        self.transformation_matrix_path = transformation_matrix_path
        self.mean_matrix_path = mean_matrix_path
        self.store = store
        
        if self.transformation_matrix_path == None:
            self.W = None # The PCA transformation matrix derived by training data
            self.mean = None
        else:
            self.W = np.load(self.transformation_matrix_path)
            self.mean = np.load(self.mean_matrix_path)

        assert mode in ['Train', 'Eval'], "mode should be either 'Train' or 'Eval'"
        self.mode = mode
            
    def reduce_and_reconstruct(self, data):
        '''
        pca reduction and reconstruction
        '''
        if self.normalize:
            scaler = StandardScaler()
            data = scaler.fit_transform(data)

        X_pca = self.pca.fit_transform(data) 
        
        X_reconstructed = self.pca.inverse_transform(X_pca) 
        
        return X_pca, X_reconstructed

    def pca_reduction(self, data):
        '''
        Input: data: (len(dataset), D)
        Output: X_pca: (len(dataset), n_components)
        '''
        X_pca = (data - self.mean) @ self.W # (len(dataset), n_components)
        return X_pca

    def pca_reconstruction(self, X_pca):
        '''
        Input: X_pca: (len(dataset), n_components)
        Output: X_reconstructed: (len(dataset), D)
        '''
        X_reconstructed = X_pca @ self.W.T + self.mean # (len(dataset), D)
        return X_reconstructed
    
    def calculate_information_loss(self):
        """
        calculate information loss for PCA 
        """
        explained_variance_ratio = self.pca.explained_variance_ratio_
        loss_ratio = 1 - np.sum(explained_variance_ratio)
        return loss_ratio

