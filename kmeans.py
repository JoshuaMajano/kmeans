import numpy as np
from PIL import Image

def kmeans(X:np.ndarray, k:int, centroids=None, max_iter=30, tolerance=1e-2):
    if centroids == None:
        k_idx = np.random.choice(len(X), k, replace=False)
        centroids = X[k_idx, :]
        distances = [[np.linalg.norm(row - k_row) for k_row in centroids] for row in X]
        labels = np.argmin(distances, axis=1)
        
        for _ in range(max_iter):
            old_centroids = centroids
            centroids = np.array([X[labels == idx].mean(axis=0) for idx in range(k)])
            distances = [[np.linalg.norm(row - k_row) for k_row in centroids] for row in X]
            labels = np.argmin(distances, axis=1)
            if np.mean(np.linalg.norm(old_centroids - centroids, axis=1)) < tolerance: 
                break
                
    elif centroids == 'kmeans++':
        idx = np.random.choice(len(X), 1)
        centroids = X[idx, :]
        for _ in range(k-1):
            next_cluster = [min([np.linalg.norm(row - k_row) for k_row in centroids]) for row in X]
            cluster_idx = np.argmax(next_cluster)
            centroids = np.vstack((centroids, X[cluster_idx, :]))
        distances = [[np.linalg.norm(row - k_row) for k_row in centroids] for row in X]
        labels = np.argmin(distances, axis=1)
        
        for _ in range(max_iter):
            old_centroids = centroids
            centroids = np.array([X[labels == idx].mean(axis=0) for idx in range(k)])
            distances = [[np.linalg.norm(row - k_row) for k_row in centroids] for row in X]
            labels = np.argmin(distances, axis=1)
            if np.mean(np.linalg.norm(old_centroids - centroids, axis=1)) < tolerance: 
                break
    return centroids, labels


def image2pixels(image_file):
    im = Image.open(image_file)
    im_array = np.asarray(im)
    if len(im_array.shape) == 2:
        h, w = im_array.shape
        dim = (h, w)
        pix = im_array.reshape((h * w),1)
    if len(im_array.shape) == 3:
        h, w, d = im_array.shape
        dim = (h, w, d)
        pix = im_array.reshape((h * w), d)
    return pix, dim
