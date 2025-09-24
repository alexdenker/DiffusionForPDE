
import matplotlib.pyplot as plt
import torch 

from skimage import data, transform
import matplotlib.pyplot as plt

B, C_in, H, W = 1, 1, 128, 128
C_out = 6

image = data.camera() 
image_128 = transform.resize(image, (H, W), anti_aliasing=True)


x = torch.from_numpy(image_128).float().reshape(1,1,H,W)

x_ft = torch.fft.rfft2(x, norm="forward")

modes = 10 

A1 = torch.randn(C_in, C_out, modes, modes, 2 )
A2 = torch.randn(C_in, C_out, modes, modes, 2 )

def compl_mul2d(input, weights):
    # Multiply input (complex) with weights (complex)
    return torch.einsum("bixy,ioxy->boxy", input, torch.view_as_complex(weights))


out_ft = torch.zeros(
    B, C_out, H, W//2 + 1,
    dtype=torch.cfloat, device=x.device
    )

out_ft2 = torch.zeros(
    B, C_out, H, W//2 + 1,
    dtype=torch.cfloat, device=x.device
    )

out_ft[:, :, :modes, :modes] = compl_mul2d(x_ft[:, :, :modes, :modes], A1)

out_ft2[:, :, :modes, :modes] = compl_mul2d(x_ft[:, :, :modes, :modes], A1)
out_ft2[:, :, -modes:, :modes] = compl_mul2d(x_ft[:, :, -modes:, :modes], A2)

out_ift = torch.fft.irfft2(out_ft, s=(H, W), norm="forward")
out_ift2 = torch.fft.irfft2(out_ft2, s=(H, W), norm="forward")

print(x_ft.shape, out_ift.shape)

#plt.figure()
#plt.imshow(torch.log(torch.abs(x_ft[0,0])+1e-6).cpu().numpy(), cmap='jet', interpolation="nearest")
#plt.colorbar()
#plt.title("FFT of random noise")
#plt.axis('image')
#plt.show()

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(9,5))
im1 = ax1.imshow(image_128, cmap='gray', interpolation="nearest")
ax1.set_title("Original Image")
ax1.axis('image')
ax1.axis("off") 

ax2.set_title(f"Reconstructed from {modes**2} modes")
im2 = ax2.imshow(out_ift[0,0].cpu().numpy(), cmap='gray', interpolation="nearest")
ax2.axis('image')       
ax2.axis("off")

ax3.set_title(f"Reconstructed from {modes**2} modes (pos+neg)")
im2 = ax3.imshow(out_ift2[0,0].cpu().numpy(), cmap='gray', interpolation="nearest")
ax3.axis('image')       
ax3.axis("off")

plt.show()