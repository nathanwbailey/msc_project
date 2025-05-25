import numpy as np
import torch

def test_transformer_model(encoder_model, seq2seq_model, loss_fn, testloader, device):
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

    seq2seq_model.eval()
    encoder_model.eval()
    test_loss = []
    with torch.no_grad():
        for data in testloader:
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
            test_loss.append(loss.item())


    print(f'Test Loss: {np.mean(test_loss):.10f}')
    return np.mean(test_loss)
