import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

"""---Make Data---"""
n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data, 1)
y0 = torch.zeros(100)
x1 = torch.normal(-2*n_data, 1)
y1 = torch.ones(100)

x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1), ).type(torch.LongTensor)

"""---Data Plot---"""
# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1],
#             c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
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
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return x

net = Net(n_feature=2, n_hidden=10, n_output=2)

"""---Net Structure---"""
# print(net)

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
loss_func = torch.nn.CrossEntropyLoss()

"""---Visualisation P1---"""
plt.ion()
plt.show()

for t in range(300):
    out = net(x)
    loss = loss_func(out, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    """---Visualisation P2---"""
    if t % 2 ==0:
        plt.cla()
        prediction = torch.max(F.softmax(out, dim=0), 1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1],
                    c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y)/200
        plt.text(1, -4, "Accuracy=%.2f" % accuracy,
                fontdict={"size": 15, "color": "blue"})
        plt.pause(0.1)

"""---Visualisation P3 (if hold)---"""
# plt.ioff()
# plt.show()
