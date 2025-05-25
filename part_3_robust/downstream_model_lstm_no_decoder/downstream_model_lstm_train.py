import numpy as np
import torch

def train_lstm_model(num_epochs, encoder_model, seq2seq_model, loss_fn, optimizer, scheduler, trainloader, validloader, device, model_save_path='lstm_model.pth'):
    def prepare_inputs(input_data, output_data, encoder_model):
        B, T, C, H, W = input_data.shape
        input_data = input_data.reshape(B*T, C, H, W)
        input_encoded_data = encoder_model(input_data)[0]
        input_encoded_data = input_encoded_data.reshape(B, T, -1)
        B, T, C, H, W = output_data.shape
        output_data = output_data.reshape(B*T, C, H, W)
        output_encoded_data = encoder_model(output_data)[0]
        output_encoded_data = output_encoded_data.reshape(B, T, -1)
        return input_encoded_data, output_encoded_data

    train_loss_plot = []
    valid_loss_plot = []
    epsilon_min = 0.1
    epsilon_initial = 1.0
    for epoch in range(num_epochs):
        epsilon = epsilon_min + (epsilon_initial - epsilon_min) * (1 - ((epoch+1) / num_epochs))
        seq2seq_model.train()
        train_loss = []
        for data in trainloader:
            optimizer.zero_grad()
            input_data = data[0].to(device)
            output_data = data[1].to(device)
            input_encoded_data, output_encoded_data = prepare_inputs(input_data, output_data, encoder_model)
            model_pred = seq2seq_model(input_encoded_data, output_encoded_data, epsilon)
            loss = loss_fn(model_pred, output_encoded_data)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        
        seq2seq_model.eval()
        valid_loss = []
        with torch.no_grad():
            for data in validloader:
                input_data = data[0].to(device)
                output_data = data[1].to(device)
                input_encoded_data, output_encoded_data = prepare_inputs(input_data, output_data, encoder_model)
                model_pred = seq2seq_model(input_encoded_data)
                loss = loss_fn(model_pred, output_encoded_data)
                valid_loss.append(loss.item())
        torch.save(seq2seq_model, model_save_path)
        print(f'Epoch: {epoch}, Train Loss: {np.mean(train_loss):.5f}, Valid Loss: {np.mean(valid_loss):.5f}, EPS: {epsilon}')
        train_loss_plot.append(np.mean(train_loss))
        valid_loss_plot.append(np.mean(valid_loss))
        scheduler.step(np.mean(valid_loss))
    return train_loss_plot, valid_loss_plot
