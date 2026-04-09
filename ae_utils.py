from pathlib import Path

from train_cvae import CVAE
# from train import AE
# from train_vae import VAE
import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import torch.nn.functional as F



def _encode(target_class: int,
            num_classes: int,
            input_dim: int,
            latent_dim: int,
            checkpoint_path: Path,
            dataset: str = 'mnist',
            X_train_tensor: torch.Tensor = None,
            y_train: torch.Tensor = None):
        """
        add description
        """
        model = CVAE(input_dim=input_dim, latent_dim=latent_dim, dataset=dataset, class_size=num_classes)
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()  # set to evaluation mode
        
        labels_tensor = torch.tensor(y_train, dtype=torch.long)
        one_hot_labels = F.one_hot(labels_tensor, num_classes=num_classes).float()

        # mask target class
        assert isinstance(target_class, int)
        mask = y_train == target_class
        X_input = X_train_tensor[mask]
        c_ = one_hot_labels[mask]

        with torch.no_grad():
            x_enc = model.cnn(X_input) if model.cnn is not None else X_input
            # after the cnn, concat x with the condition
            h = model.encoder(torch.cat([x_enc, c_], 1))
            mu_ = model.fc21(h)
        return mu_

def _decode(mu_: torch.Tensor,
            target_class: int,
            num_classes: int,
            input_dim: int,
            latent_dim: int,
            checkpoint_path: Path,
            dataset: str = 'mnist'):
    """
    add description
    """
    model = CVAE(input_dim=input_dim, latent_dim=latent_dim, dataset=dataset, class_size=num_classes)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()  # set to evaluation mode
    
    c_target = F.one_hot(torch.tensor([target_class]), num_classes=num_classes).float()
    
    with torch.no_grad():
        batch_size = mu_.shape[0]
        c_expanded = c_target.repeat(batch_size, 1)
        dec_inputs = torch.cat([mu_, c_expanded], 1)
        recon = model.decoder(dec_inputs)
    return recon


    
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
