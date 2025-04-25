# model.py

import torch
import torch.nn as nn
import os
import joblib

class LSTMModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=256, output_size=9):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)  # 1 layer LSTM
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return out

def load_model(model_path='trained_model.pth', scaler_x_path='scaler_x.pkl', scaler_y_path='scaler_y.pkl'):
    input_size = 4  # hour, day, month, weekday
    hidden_size = 256
    output_size = 9

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel(input_size, hidden_size, output_size).to(device)

    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load scalers
    scaler_x = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)

    return model, scaler_x, scaler_y
