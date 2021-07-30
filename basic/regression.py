import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())

"""---Data Plot---"""
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

class Net(torch.nn.Module):
    """
    Net (
      (hidden): Linear (n_feature -> n_hidden)
      (predict): Linear (n_hidden -> n_output)
    )
    """
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

# net = Net(n_feature=1, n_hidden=10, n_output=1)
#
# """---Net Structure---"""
# # print(net)
#
# optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
# loss_func = torch.nn.MSELoss()
#
# """---Visualisation P1---"""
# # plt.ion()
# # plt.show()
#
# for _ in range(300):
#     prediction = net(x)
#     loss = loss_func(prediction, y)
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     """---Visualisation P2---"""
#     # if t % 5 ==0:
#     #     plt.cla()
#     #     plt.scatter(x.data.numpy(), y.data.numpy())
#     #     plt.plot(x.data.numpy(), prediction.data.numpy(), "r-", lw=3)
#     #     plt.text(0.5, 0, "Loss=%.4f" % loss.data.numpy(),
#     #             fontdict={"size": 15, "color": "blue"})
#     #     plt.pause(0.1)

"""---Experiment on hidden units---"""
loss_comparison = []
for i in range(1, 21):
    loss_trial = []
    for j in range(5):
        net = Net(n_feature=1, n_hidden=i, n_output=1)

        """---Net Structure---"""
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        loss_func = torch.nn.MSELoss()

        for _ in range(100):
            prediction = net(x)
            loss = loss_func(prediction, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_trial.append(loss.data.numpy())
    loss_comparison.append(min(loss_trial))

plt.scatter(range(len(loss_comparison)), loss_comparison)
plt.show()
