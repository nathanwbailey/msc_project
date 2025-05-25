import numpy as np
import torch

def train_transformer_model(num_epochs, encoder_model, seq2seq_model, loss_fn, optimizer, scheduler, trainloader, validloader, device, model_save_path='downstream_model_no_decoder.pth'):
    def prepare_inputs(input_data, output_data, encoder_model):
        B, T, C, H, W = input_data.shape
        input_data = input_data.reshape(B*T, C, H, W)
        input_encoded_data = encoder_model(input_data)[0]
        input_encoded_data = input_encoded_data.reshape(B, T, -1)
        B, T, C, H, W = output_data.shape
        output_data = output_data.reshape(B*T, C, H, W)
        output_encoded_data = encoder_model(output_data)[0]
        output_encoded_data = output_encoded_data.reshape(B, T, -1)
        output_encoded_data = torch.cat((input_encoded_data[:, -1, ...].unsqueeze(1), output_encoded_data), dim=1)
        return input_encoded_data, output_encoded_data

    train_loss_plot = []
    valid_loss_plot = []
    for epoch in range(num_epochs):
        seq2seq_model.train()
        train_loss = []
        for data in trainloader:
            optimizer.zero_grad()
            input_data = data[0].to(device)
            output_data = data[1].to(device)
            input_encoded_data, output_encoded_data = prepare_inputs(input_data, output_data, encoder_model)
            target_input = output_encoded_data[:, :-1, ...]
            target_output = output_encoded_data[:, 1:, ...]
            T = target_input.shape[1]
            # target_mask = torch.triu(torch.ones(T, T) * float('-inf'), diagonal=1).to(device)
            target_mask = torch.nn.Transformer.generate_square_subsequent_mask(T)
            model_pred = seq2seq_model(input_encoded_data, target_input, tgt_mask=target_mask, tgt_is_causal=True)
            loss = loss_fn(model_pred, target_output)
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

                target_output = output_encoded_data[:, 1:, ...]
                model_pred = input_encoded_data[:, -1, :].unsqueeze(1)
                num_steps = output_encoded_data.shape[1] - 1 
                for _ in range(num_steps):
                    T_cur = model_pred.size(1)
                    tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(T_cur).to(device)
                    pred = seq2seq_model(input_encoded_data, model_pred, tgt_mask=tgt_mask, tgt_is_causal=True)
                    next_token = pred[:, -1, :].unsqueeze(1)
                    model_pred = torch.cat([model_pred, next_token], dim=1)

                model_pred = model_pred[:, 1:, ...]
                loss = loss_fn(model_pred, target_output)
                valid_loss.append(loss.item())
        lr = optimizer.param_groups[0]['lr']
        torch.save(seq2seq_model, model_save_path)
        print(f'Epoch: {epoch}, Train Loss: {np.mean(train_loss):.10f}, Valid Loss: {np.mean(valid_loss):.10f}, LR: {lr}')
        train_loss_plot.append(np.mean(train_loss))
        valid_loss_plot.append(np.mean(valid_loss))
        scheduler.step(np.mean(valid_loss))
    return train_loss_plot, valid_loss_plot
