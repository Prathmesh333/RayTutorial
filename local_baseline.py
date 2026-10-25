"""Plain PyTorch baseline; no Ray runtime or process group required."""
from pathlib import Path
import torch


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.manual_seed(7)
    x = torch.linspace(-1, 1, 1024).reshape(-1, 1)
    y = 3 * x + 2
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(8):
        for xb, yb in zip(x.split(32), y.split(32)):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            optimizer.step()
    with torch.inference_mode():
        mse = torch.nn.functional.mse_loss(model(x), y).item()
        predictions = model(torch.tensor([[0.0], [1.0]])).reshape(-1).tolist()
    assert mse < 0.05
    Path("artifacts").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "artifacts/model.pt")
    print({"mse": mse, "weights": model.weight.item(), "bias": model.bias.item()})
    print({"input": [0.0, 1.0], "prediction": predictions})
