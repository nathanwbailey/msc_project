import numpy as np
import torch


def test_lstm_model(
    encoder_model, seq2seq_model, loss_fn, testloader, device
):
    def prepare_inputs(input_data, output_data, encoder_model):
        B, T, C, H, W = input_data.shape
        input_data = input_data.reshape(B * T, C, H, W)
        input_encoded_data = encoder_model(input_data)[0]
        input_encoded_data = input_encoded_data.reshape(B, T, -1)
        B, T, C, H, W = output_data.shape
        output_data = output_data.reshape(B * T, C, H, W)
        output_encoded_data = encoder_model(output_data)[0]
        output_encoded_data = output_encoded_data.reshape(B, T, -1)
        return input_encoded_data, output_encoded_data

    seq2seq_model.eval()
    test_loss = []
    with torch.no_grad():
        for data in testloader:
            input_data = data[0].to(device)
            output_data = data[1].to(device)
            input_encoded_data, output_encoded_data = prepare_inputs(
                input_data, output_data, encoder_model
            )
            model_pred = seq2seq_model(input_encoded_data)
            loss = loss_fn(model_pred, output_encoded_data)
            test_loss.append(loss.item())

    print(f"Test Loss: {np.mean(test_loss):.5f}")
    return np.mean(test_loss)
