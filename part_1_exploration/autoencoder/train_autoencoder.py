import torch
import torch.nn.functional as F




def train_autoencoder(model, num_epochs, trainloader, testloader, loss_fn, optimizer, scheduler, device, model_save_path="det_autoencoder.pth"):
    train_losses = []
    test_losses = []
    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        for data in trainloader:
            optimizer.zero_grad()
            data = data.to(device)
            recon_data = model(data)
            loss_batch = loss_fn(recon_data, data)
            loss_batch.backward()
            optimizer.step()
            total_loss += loss_batch.item()

                
        train_losses.append([total_loss / len(trainloader)])

        total_loss = 0
        model.eval()
        with torch.no_grad():
            for data in testloader:
                data = data.to(device)
                recon_data = model(data)
                loss_batch =  loss_fn(recon_data, data)
                total_loss += loss_batch.item()
        test_losses.append([total_loss / len(testloader)])
        lr = optimizer.param_groups[0]['lr']
        print(f'Epoch: {epoch}, train loss = {train_losses[-1][0]:.2f}, test loss = {test_losses[-1][0]:.2f}, lr = {lr:.5f}')
        torch.save(model, model_save_path)

        scheduler.step(test_losses[-1][0])
    return train_losses, test_losses
