from pathlib import Path

from train_cvae import CVAE
# from train import AE
# from train_vae import VAE
import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd



## encode ##
def _construct_latent_vectors_list(
    checkpoint_path: Path,
    X_training_tensor: torch.Tensor,
    input_dim: int,
    latent_dim: int, 
    ae_type: int,
    c: torch.Tensor = None,
    dataset: str = 'adima',
    class_size: int = 4):
    """
    extract the full set of latent vectors for X_training_tensor.

    args:
        checkpoint_path (str) (autoencoder .pth file)  
        X_training_tensor (Tensor)  
        input_dim (int) (encoder input size)  
        latent_dim (int) (bottleneck size)  
        ae_type (int): 0=AE, 1=VAE, 2=VAE with external classes, 3=CVAE
        c (Tensor, optional): condition/class tensor for CVAE (one-hot encoded)
        dataset (str): dataset type for CVAE ('adima' or 'mnist')
        class_size (int): number of classes for CVAE

    return:
        latent_vectors (list containing the latent vector for every sample)
    """
    if ae_type == 3:
        # Conditional VAE
        model = CVAE(input_dim=input_dim, latent_dim=latent_dim, dataset=dataset, class_size=class_size)
        if c is None:
            raise ValueError("CVAE requires condition tensor 'c' (one-hot encoded classes)")
    elif ae_type == 0:
        from agile_ima_ml.autoencoders.train import AE
        model = AE(input_dim=input_dim, latent_dim=latent_dim)
    elif ae_type == 1:
        from agile_ima_ml.autoencoders.train_vae import VAE
        model = VAE(input_dim=input_dim, latent_dim=latent_dim)
    elif ae_type == 2:
        # VAE with external conditions (treated as regular VAE for encoding)
        from agile_ima_ml.autoencoders.train_vae import VAE
        model = VAE(input_dim=input_dim, latent_dim=latent_dim)
    else:
        raise Exception(f"Unknown ae_type: {ae_type}")
    
    print(f"loading model from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()  # set to evaluation mode
    
    with torch.no_grad():
        latent_vectors = []
        for i in range(len(X_training_tensor)):
            sample = X_training_tensor[i]
            
            if ae_type == 3:
                # CVAE: needs both sample and condition
                sample = sample.unsqueeze(0)
                condition = c[i].unsqueeze(0) if len(c[i].shape) == 1 else c[i:i+1]
                _, mu, _ = model(sample, condition)
                latent_sample = mu.squeeze(0)
            elif ae_type == 2:
                # VAE with external conditions - use only the latent from VAE
                sample = sample.unsqueeze(0)
                _, _, mu, _ = model(sample)
                latent_sample = mu.squeeze(0)
            elif ae_type == 1:
                #  VAE
                sample = sample.unsqueeze(0)
                _, _, mu, _ = model(sample)
                latent_sample = mu.squeeze(0)
            else:
                #  AE
                latent_sample = model.encoder(sample)
            
            latent_vectors.append(latent_sample)
        
        # to numpy
        latent_vectors = torch.stack(latent_vectors).numpy()
        return latent_vectors
  
    
def _plot_latent_space(latent_vectors, y_train, show=False):
    """
    plot the 1 or 2D latent space of the autoencoder
    """
    if not show:
        return
    if latent_vectors.shape[1] == 1:
        print("1D latent space")
        plt.scatter(latent_vectors[:, 0], np.zeros_like(latent_vectors[:, 0]), c=y_train, cmap='viridis', s=5)
        plt.xlabel('Latent Dimension 1')
    elif latent_vectors.shape[1] == 2:
        print("2D latent space")
        plt.scatter(latent_vectors[:, 0], latent_vectors[:, 1], c=y_train, cmap='viridis', s=5)
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
    else:
        raise ValueError("Latent vectors must be 1D or 2D for plotting")
    
    plt.title('Latent Space Representation')

#    os.makedirs(root/'data/adima/artificial/', exist_ok=True)
#    plt.savefig(root/'data/adima/artificial/latent_space_plot.png', dpi=300)
    #plt.show()

def _save_vectors_to_csv(vectors, y_train, output_path, filename, save=False):
    """
    save the full/latent vectors to a csv file
    """
    if not save:
        return
    df = pd.DataFrame(vectors, columns=[f'dim_{i+1}' for i in range(vectors.shape[1])])
    df.insert(0, 'Spannung', y_train)
    df.to_csv(os.path.join(output_path, filename), index=False)
    print(f"Latent vectors saved to {output_path} as {filename}")


## decode ##
def _reconstuct_from_latent_vectors_list(
    checkpoint_path: str,
    latent_vectors: torch.Tensor,
    input_dim: int, # the reconstruction will have this size
    latent_dim: int):
    """
    reconstruct the samples from the latent vectors list.
    args:
        checkpoint_path (str) (autoencoder .pth file)  
        latent_vectors (Tensor)  
        input_dim (int) (encoder input size)  
        latent_dim (int) (bottleneck size)  

    return:
        decoded vectors list from a latent vector list
    """
    AE_loaded = AE(input_dim=input_dim, latent_dim=latent_dim)
    AE_loaded.load_state_dict(torch.load(checkpoint_path))
    AE_loaded.eval()  # set to evaluation mode
    with torch.no_grad():
        recon_vectors = []
        for i in range(len(latent_vectors)):
            sample = latent_vectors[i]
            sample = torch.tensor(sample, dtype=torch.float32)
            recon_sample = AE_loaded.decoder(sample)
            recon_vectors.append(recon_sample)
        # to numpy
        recon_vectors = torch.stack(recon_vectors).numpy()
        return recon_vectors
    
# sampling

def _bounds(latent, mask):
    """
    min & max per dimension for the chosen class (mask is boolean)
    the function should be able to handle both numpy array and pytorch tensors
    same way of indexing and same min max methods
    """
    block = latent[mask]
    return block.min(0), block.max(0)          

def _sample_box_points(vmin, vmax, n, latent_dim=1):
    """sample n points uniformly in the bounding box"""
        # for the 1D case a "1D" circle would be sufficient
        # 2D case: sample in the rectangle
    if latent_dim == 1:
        x_samples = np.random.uniform(vmin[0], vmax[0], n)
        return np.c_[x_samples]
    if latent_dim == 2:
        x_samples = np.random.uniform(vmin[0], vmax[0], n)    
        y_samples = np.random.uniform(vmin[1], vmax[1], n)
        return np.c_[x_samples, y_samples]

def _sample_diag_points(vmin, vmax, n=200):
    """diagonal from (vmin,x) to (vmax,y)"""
    t = np.linspace(0, 1, n)
    return np.c_[vmin[0] + t*(vmax[0]-vmin[0]),
                 vmax[1] - t*(vmax[1]-vmin[1])]

def _sample_circle_points(vmin, vmax, n, latent_dim=1):
    """sample n points uniformly inside a circle"""
    if latent_dim == 1:
        # 1D case: sample along the diameter
        center = (vmin[0] + vmax[0]) / 2
        radius = (vmax[0] - vmin[0]) / 2
        return np.random.uniform(center - radius, center + radius, (n, 1))
    else:
        # 2D case: sample inside circle using polar coordinates
        center_x = (vmin[0] + vmax[0]) / 2
        center_y = (vmin[1] + vmax[1]) / 2
        radius = np.sqrt(((vmax[0] - vmin[0])/2)**2 + ((vmax[1] - vmin[1])/2)**2)
        
        # Sample using polar coordinates for uniform distribution
        r = radius * np.sqrt(np.random.uniform(0, 1, n))
        theta = np.random.uniform(0, 2*np.pi, n)
        
        x = center_x + r * np.cos(theta)
        y = center_y + r * np.sin(theta)
        return np.c_[x, y]
