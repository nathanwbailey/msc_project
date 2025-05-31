import numpy as np
import torch


def train_model(
    model,
    num_epochs,
    trainloader,
    testloader,
    optimizer,
    scheduler,
    device,
    loss_fn,
    cycle_loss,
    model_save_path="barlow_twins.pth",
):
    """
    Train a model with contrastive and cycle consistency losses.
    """
    for epoch in range(num_epochs):
        train_loss, train_cycle_loss = [], []
        con_train_loss = []
        valid_loss, valid_cycle_loss = [], []
        con_valid_loss = []

        model.train()
        for data in trainloader:
            optimizer.zero_grad()
            X = data[0].to(device)
            X_prime = data[1].to(device)
            X_prime_2 = data[2].to(device)
            B, T, C, H, W = X.shape
            X = X.reshape(B * T, C, H, W)
            B, T, S, C, H, W = X_prime.shape
            X_prime = X_prime.reshape(B * T * S, C, H, W)
            X_prime_2 = X_prime_2.reshape(B * T * S, C, H, W)

            Z = model(X)
            Z_prime = model(X_prime)
            Z_prime_2 = model(X_prime_2)

            Z_prime_cyc = Z_prime.reshape(B * T, S, -1)
            Z_prime_2_cyc = Z_prime_2.reshape(B * T, S, -1)

            N = X.shape[0]
            embeddings = torch.cat((Z, Z_prime, Z_prime_2), dim=0)
            labels = torch.cat([
                torch.arange(N),
                torch.arange(N).repeat_interleave(S),
                torch.arange(N).repeat_interleave(S)
            ], dim=0)


            cyc_loss = torch.tensor(0.0, device=X.device)
            for i in range(S):
                cyc_loss += cycle_loss(Z_prime_cyc[:, i] - 2 * Z + Z_prime_2_cyc[:, i], torch.zeros_like(Z))
            cyc_loss = cyc_loss / S

            loss = loss_fn(embeddings, labels)
            loss_batch = loss + cyc_loss
            loss_batch.backward()
            optimizer.step()

            con_train_loss.append(loss.item())
            train_cycle_loss.append(cyc_loss.item())
            train_loss.append(loss_batch.item())

        model.eval()
        with torch.no_grad():
            for data in testloader:
                X = data[0].to(device)
                X_prime = data[1].to(device)
                X_prime_2 = data[2].to(device)
                B, T, C, H, W = X.shape
                X = X.reshape(B * T, C, H, W)
                B, T, S, C, H, W = X_prime.shape
                X_prime = X_prime.reshape(B * T * S, C, H, W)
                X_prime_2 = X_prime_2.reshape(B * T * S, C, H, W)

                Z = model(X)
                Z_prime = model(X_prime)
                Z_prime_2 = model(X_prime_2)

                Z_prime_cyc = Z_prime.reshape(B * T, S, -1)
                Z_prime_2_cyc = Z_prime_2.reshape(B * T, S, -1)

                N = X.shape[0]
                embeddings = torch.cat((Z, Z_prime, Z_prime_2), dim=0)
                labels = torch.cat([
                    torch.arange(N),
                    torch.arange(N).repeat_interleave(S),
                    torch.arange(N).repeat_interleave(S)
                ], dim=0)


                cyc_loss = torch.tensor(0.0, device=X.device)
                for i in range(S):
                    cyc_loss += cycle_loss(Z_prime_cyc[:, i] - 2 * Z + Z_prime_2_cyc[:, i], torch.zeros_like(Z))
                cyc_loss = cyc_loss / S

                loss = loss_fn(embeddings, labels)
                loss_batch = loss + cyc_loss

                con_valid_loss.append(loss.item())
                valid_cycle_loss.append(cyc_loss.item())
                valid_loss.append(loss_batch.item())

        torch.save(model, model_save_path)
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch: {epoch}\n"
            f"  Train Loss:        {np.mean(train_loss):.4f}\n"
            f"    Contrastive:   {np.mean(con_train_loss):.4f}\n"
            f"    Cycle:           {np.mean(train_cycle_loss):.4f}\n"
            f"  Valid Loss:        {np.mean(valid_loss):.4f}\n"
            f"    Contrastive:   {np.mean(con_valid_loss):.4f}\n"
            f"    Cycle:           {np.mean(valid_cycle_loss):.4f}\n"
            f"  LR: {lr:.2e}"
        )
        scheduler.step(np.mean(valid_loss))
