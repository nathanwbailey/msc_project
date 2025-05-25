import torch
import numpy as np
import torch.nn.functional as F

def train_model(model, num_epochs, trainloader, testloader, optimizer, device, loss_fn, model_save_path="sup_con_model.pth"):
    for epoch in range(num_epochs):
        train_loss = []
        valid_loss = []
        model.train()
        sim_cosine_list = []
        val_sim_cosine_list = []
        rand_val_sim_cosine_list = []
        rand_sim_cosine_list = []
        for data in trainloader:
            optimizer.zero_grad()
            X = data[0].to(device)
            X_prime = data[1].to(device)
            Y = data[2].to(device)
            X_embeddings = model(X)
            X_prime_embeddings = model(X_prime)
            Y = torch.cat((Y, Y), dim=0)
            embeddings = torch.cat((X_embeddings, X_prime_embeddings), dim=0)
            loss_batch = loss_fn(embeddings, Y)
            loss_batch.backward()
            optimizer.step()
            sim_cosine = F.cosine_similarity(X_embeddings, X_prime_embeddings, dim=1)

            rand_indices = torch.randperm(X_embeddings.shape[0])
            sim_cosine = F.cosine_similarity(X_embeddings, X_prime_embeddings[rand_indices], dim=1)
            rand_sim_cosine_list.append(sim_cosine.mean().item())

            sim_cosine_list.append(sim_cosine.mean().item())
            train_loss.append(loss_batch.item())

        model.train()
        with torch.no_grad():
            for data in testloader:
                X = data[0].to(device)
                X_prime = data[1].to(device)
                Y = data[2].to(device)
                X_embeddings = model(X)
                X_prime_embeddings = model(X_prime)
                Y = torch.cat((Y, Y), dim=0)
                embeddings = torch.cat((X_embeddings, X_prime_embeddings), dim=0)

                sim_cosine = F.cosine_similarity(X_embeddings, X_prime_embeddings, dim=1)
                val_sim_cosine_list.append(sim_cosine.mean().item())

                rand_indices = torch.randperm(X_embeddings.shape[0])
                sim_cosine = F.cosine_similarity(X_embeddings, X_prime_embeddings[rand_indices], dim=1)
                rand_val_sim_cosine_list.append(sim_cosine.mean().item())

                loss_batch = loss_fn(embeddings, Y)
                valid_loss.append(loss_batch.item())

        torch.save(model, model_save_path)
        # print(f'Avg Train Sim Cosine: {np.mean(sim_cosine_list):.2f}')
        # print(f'Avg Rand Train Sim Cosine: {np.mean(rand_sim_cosine_list):.2f}')
        # print(f'Avg Valid Sim Cosine: {np.mean(val_sim_cosine_list):.2f}')
        # print(f'Avg Rand Valid Sim Cosine: {np.mean(rand_val_sim_cosine_list):.2f}')
        # print('\n')
        print(f'Epoch: {epoch}, Train Loss: {np.mean(train_loss):.2f}, Validation Loss: {np.mean(valid_loss):.2f}')