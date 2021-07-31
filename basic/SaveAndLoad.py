import torch
import matplotlib.pyplot as plt

"""---Make Data---"""
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())

"""---Save And Load Models---"""
def save():
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    optimizer = torch.optim.SGD(net1.parameters(), lr=0.5)
    loss_func = torch.nn.MSELoss()

    for _ in range(100):
        prediction = net1(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(net1, "net.pkl")  # Save the entire net
    torch.save(net1.state_dict(), "net_params.pkl") # Only save the parameters

    # Plot net1 result
    plt.figure(1, figsize=(10, 3))
    plt.subplot(131)
    plt.title("Net 1")
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=3)

def restore_net():
    net2 = torch.load("net.pkl")
    prediction = net2(x)

    # Plot net2 result
    plt.subplot(132)
    plt.title("Net 2")
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=3)

def restore_params():
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    net3.load_state_dict(torch.load("net_params.pkl"))
    prediction = net3(x)

    # Plot net2 result
    plt.subplot(133)
    plt.title("Net 3")
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=3)
    plt.show()

save()
restore_net()
restore_params()

plt.scatter(x.data.numpy(), y.data.numpy())
plt.plot(x.data.numpy(), prediction.data.numpy(), "r-", lw=3)
plt.text(0.5, 0, "Loss=%.4f" % loss.data.numpy(),
        fontdict={"size": 15, "color": "blue"})
plt.pause(0.1)
