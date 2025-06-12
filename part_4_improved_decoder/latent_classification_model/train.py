import numpy as np
import torch


def train_classification_model(
    model,
    encoder_model,
    num_epochs,
    trainloader,
    testloader,
    device,
    optimizer,
    scheduler,
    loss_fn,
    model_save_path="latent_model.pth",
):
    for epoch in range(1, num_epochs):
        train_losses = []
        valid_losses = []
        model.train()
        for batch in trainloader:
            optimizer.zero_grad()
            x = batch[0].to(device)
            y = batch[1].to(device)
            encoded_x = encoder_model(x)[0]
            pred = model(encoded_x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            for batch in testloader:
                x = batch[0].to(device)
                y = batch[1].to(device)
                encoded_x = encoder_model(x)[0]
                pred = model(encoded_x)
                loss = loss_fn(pred, y)
                valid_losses.append(loss.item())

        torch.save(model, model_save_path)
        scheduler.step(np.mean(valid_losses))
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch: {epoch}, Train Loss: {np.mean(train_losses):.4f}, Valid Loss: {np.mean(valid_losses):.4f}, LR: {lr}"
        )


def test_classification_network(
    model, encoder_model, device, loss_fn, testloader
):
    test_losses = []
    model.eval()
    with torch.no_grad():
        for batch in testloader:
            x = batch[0].to(device)
            y = batch[1].to(device)
            encoded_x = encoder_model(x)[0]
            pred = model(encoded_x)
            loss = loss_fn(pred, y)
            test_losses.append(loss.item())

    print(f"Test Loss: {np.mean(test_losses):.4f}")
