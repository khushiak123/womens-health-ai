# python
import torch
import syft as sy

# Initialize Syft Virtual Worker
alice = sy.VirtualMachine(name="alice")
alice_client = alice.get_root_client()

# Data stored on Alice's machine
remote_data = torch.tensor([[20.0, 22.0, 24.0], [30.0, 32.0, 34.0]])
remote_data_ptr = remote_data.send(alice_client)

# Simple Model
model = torch.nn.Linear(3, 1)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for _ in range(10):
    optimizer.zero_grad()
    predictions = model(remote_data_ptr.get())
    loss = loss_fn(predictions, torch.tensor([[1.0], [2.0]]))
    loss.backward()
    optimizer.step()

print("Federated learning training complete.")