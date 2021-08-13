import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import numpy as np
import matplotlib.pyplot as plt

"""--- Hyper-parameters ---"""
EPOCH = 10
BATCH_SIZE = 64
LR = 0.005
DOWNLOAD_MNIST = False
N_TEST_IMG = 5

"""--- Manage Data ---"""
train_data = torchvision.datasets.MNIST(
    root="../data/mnist/",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)

# print(train_data.data.size())
# print(train_data.targets.size())
# plt.imshow(train_data.data[2].numpy(), cmap="gray")
# plt.title("%i" % train_data.targets[2])
# plt.show()

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, 3)  # compress to 3 features for visualization
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28*28),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

autoencoder = AutoEncoder()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()

"""--- visualization P1 ---"""
# fig, ax = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
# plt.ion()
# view_data = train_data.train_data[:N_TEST_IMG].view(-1, 28*28).type(torch.FloatTensor)/255.
# for i in range(N_TEST_IMG):
#     ax[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap="gray")
#     ax[0][i].set_xticks(())
#     ax[0][i].set_yticks(())

for epoch in range(EPOCH):
    for step, (x, b_label) in enumerate(train_loader):
        b_x = x.view(-1, 28*28)
        b_y = x.view(-1, 28*28)

        encoded, decoded = autoencoder(b_x)
        loss = loss_func(decoded, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print("Epoch: ", epoch, "| train loss: %.4f" % loss.data.numpy())
        """--- visualization P2 ---"""
#         _, decoded_data = autoencoder(view_data)
#         for i in range(N_TEST_IMG):
#             ax[1][i].clear()
#             ax[1][i].imshow(np.reshape(decoded_data.data.numpy()[i], (28, 28)), cmap="gray")
#             ax[1][i].set_xticks(())
#             ax[1][i].set_yticks(())
#         plt.draw()
#         plt.pause(0.05)
#
# plt.ioff()
# plt.show()

"""--- visualization P3 (in 3D) ---"""
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

view_data = train_data.train_data[:200].view(-1, 28*28).type(torch.FloatTensor)/255.
encoded_data, _ = autoencoder(view_data)
fig = plt.figure(2)
ax = Axes3D(fig)
X, Y, Z = encoded_data.data[:, 0].numpy(), encoded_data.data[:, 1].numpy(), encoded_data.data[:, 2].numpy()
values = train_data.train_labels[:200].numpy()
for x, y, z, s in zip(X, Y, Z, values):
    c = cm.rainbow(int(255*s/9))
    ax.text(x, y, z, s, backgroundcolor=c)
ax.set_xlim(X.min(), X.max())
ax.set_ylim(Y.min(), Y.max())
ax.set_zlim(Z.min(), Z.max())
plt.show()
