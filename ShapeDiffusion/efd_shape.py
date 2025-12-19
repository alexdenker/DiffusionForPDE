"""
Compute the elliptic Fourier descriptors for 2D shapes for various numbers of bases and reconstruct the shapes from these descriptors.

"""

import torch

import matplotlib.pyplot as plt

from dataset import MNISTShapesDataset
from utils import fourier_coefficients, inverse_fourier


def plot_reconstructions():
    device = "cuda"

    num_bases_ = [2, 4, 6, 8, 10, 16]
    num_landmarks_ = [32, 64, 128]

    fig, axes = plt.subplots(len(num_landmarks_), len(num_bases_)+1, figsize=(26,8))


    for i, num_landmarks in enumerate(num_landmarks_):
        dataset = MNISTShapesDataset(class_label=3, num_landmarks=num_landmarks)

        torch.manual_seed(0)
        x = dataset[0].to(device).unsqueeze(0)

        print("Landmark shape: ", x.shape)

        axes[i,0].plot(x[0,:,0].cpu().numpy(), x[0,:,1].cpu().numpy(), '-o')
        axes[i,0].set_title('Original shape (MNIST digit 3)')

        for j, num_bases in enumerate(num_bases_):  
            coeffs = fourier_coefficients(x, num_bases=num_bases)
            print("Fourier coefficients shape: ", coeffs.shape)
            x_ift = inverse_fourier(coeffs, num_pts=num_landmarks)
            axes[i, j+1].plot(x_ift[0,:,0].detach().cpu().numpy(), x_ift[0,:,1].detach().cpu().numpy(), '-o')
            axes[i, j+1].set_title(f'Reconstructed with {num_bases} EFDs')

    plt.show()


def create_publication_figure():
    """Create a high-quality figure showing the first 4 MNIST shapes with 64 landmarks.
    Suitable for ICML paper (half-page width)."""
    
    dataset = MNISTShapesDataset(class_label=3, num_landmarks=64, train=False)
    
    # Create figure with half-page width (3.5 inches for ICML)
    fig, axes = plt.subplots(2, 4, figsize=(3.5, 1.75), dpi=300)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.05, wspace=0.1, hspace=0.2)
    
    for idx in range(8):
        pts = dataset[idx]
        ax = axes[idx // 4, idx % 4]
        
        # Plot the shape as a closed contour with markers
        ax.plot(pts[:, 0], pts[:, 1], 'o-', linewidth=1.5, markersize=2, color='#1f77b4')
        
        # Close the contour by plotting the last point back to the first
        ax.plot([pts[-1, 0], pts[0, 0]], [pts[-1, 1], pts[0, 1]], 'o-', 
               linewidth=1.5, markersize=2, color='#1f77b4')
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
        
        # Remove axes and labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
        # Add simple labels
        #ax.set_title(f'Sample {idx+1}', fontsize=8, pad=3)
    
    #fig.suptitle('MNIST Digit 3 Shapes (64 Landmarks)', fontsize=9, y=0.98)
    
    # Save as high-quality PDF for paper submission
    plt.savefig('mnist_shapes_dataset.pdf', dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.savefig('mnist_shapes_dataset.png', dpi=300, bbox_inches='tight', pad_inches=0.02)
    
    plt.show()


if __name__ == "__main__":
    create_publication_figure()

    plot_reconstructions()



