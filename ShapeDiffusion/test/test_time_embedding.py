"""
I want to test the time embedding. For every t, we get an embedding
and I want to see how the embeddings look like.
"""

import torch
import matplotlib.pyplot as plt

import sys, os 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simple_network import timestep_embedding


t = torch.linspace(0, 1, 100)

max_period = 2.0
emb = timestep_embedding(t, dim=32, max_period=max_period)

print(emb.shape)

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,6))

ax1.imshow(emb)
ax1.set_ylabel("time")
ax1.set_xlabel("embedding dimension")

for i in [0, 20, 40, 60, 80, len(t)-1]:
    ax2.plot(emb[i,:], label=f"time {t[i]:.4f}")
ax2.set_xlabel("embedding dimension")
ax2.legend()
fig.suptitle("Time-embedding")
plt.show()