import torch
import matplotlib.pyplot as plt

torch.manual_seed(1)

"""--- Hyper-parameters ---"""
N_SAMPLES = 20
N_HIDDEN = 300

"""--- Manage Data ---"""
x = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), 1)
y = x + 0.3*torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))

test_x = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), 1)
test_y = test_x + 0.3*torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))

# plt.scatter(x.data.numpy(), y.data.numpy(), c='magenta', s=50, alpha=0.5, label='train')
# plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='cyan', s=50, alpha=0.5, label='test')
# plt.legend(loc='upper left')
# plt.ylim((-2.5, 2.5))
# plt.show()

net_overfitting = torch.nn.Sequential(
    torch.nn.Linear(1, N_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN, N_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN, 1),
)

net_dropped = torch.nn.Sequential(
    torch.nn.Linear(1, N_HIDDEN),
    torch.nn.Dropout(0.5),
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN, N_HIDDEN),
    torch.nn.Dropout(0.5),
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN, 1),
)

optimizer_over = torch.optim.Adam(net_overfitting.parameters(), lr=0.01)
optimizer_drop = torch.optim.Adam(net_dropped.parameters(), lr=0.01)
loss_func = torch.nn.MSELoss()

"""--- visualization P1 ---"""
plt.ion()

for t in range(1000):
    pred_over = net_overfitting(x)
    pred_drop = net_dropped(x)

    loss_over = loss_func(pred_over, y)
    loss_drop = loss_func(pred_drop, y)

    optimizer_over.zero_grad()
    optimizer_drop.zero_grad()
    loss_over.backward()
    loss_drop.backward()
    optimizer_over.step()
    optimizer_drop.step()

    if t % 10 == 0:
        net_overfitting.eval()  # Not necessary
        net_dropped.eval()      # Important to ignore dropout during evaluation

        plt.cla()

        test_pred_over = net_overfitting(test_x)
        test_pred_drop = net_dropped(test_x)

        """--- visualization P2 ---"""
        plt.scatter(x.data.numpy(), y.data.numpy(), c='magenta', s=50, alpha=0.3, label='train')
        plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='cyan', s=50, alpha=0.3, label='test')
        plt.plot(test_x.data.numpy(), test_pred_over.data.numpy(), 'r-', lw=3, label='overfitting')
        plt.plot(test_x.data.numpy(), test_pred_drop.data.numpy(), 'b--', lw=3, label='dropout(50%)')
        plt.text(0, -1.2, 'overfitting loss=%.4f' % loss_func(test_pred_over, test_y).data.numpy(), fontdict={'size': 12, 'color':  'red'})
        plt.text(0, -1.5, 'dropout loss=%.4f' % loss_func(test_pred_drop, test_y).data.numpy(), fontdict={'size': 12, 'color': 'blue'})

        net_overfitting.train()
        net_dropped.train()     # Change it back for further training

        plt.draw()
        plt.pause(0.05)
plt.ioff()
