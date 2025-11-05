import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, output_size=1):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        if x.is_cuda:
            hidden = hidden.cuda()

        rnn_out, hidden = self.rnn(x, hidden)
        output = self.fc(rnn_out)
        # output shape: (batch_size, sequence_length, output_size)
        return output, hidden

    def predict_next(self, x, hidden=None):
        output, hidden = self.forward(x, hidden)
        prediction = output[:, -1, :]
        return prediction, hidden


if __name__ == "__main__":
    print("=" * 50)
    print("測試RNN模型")
    print("=" * 50)

    model = SimpleRNN(input_size=1, hidden_size=32, num_layers=1, output_size=1)
    print("\n模型結構：")
    print(model)

    batch_size = 2
    sequence_length = 10
    input_size = 1

    test_input = torch.randn(batch_size, sequence_length, input_size)
    print(f"\n輸入資料形狀: {test_input.shape}")
    print(f"批次大小: {batch_size}")
    print(f"序列長度: {sequence_length}")
    print(f"輸入特徵數: {input_size}")

    output, hidden = model(test_input)
    print(f"\n輸出資料形狀: {output.shape}")
    print(f"批次大小: {output.shape[0]}")
    print(f"序列長度: {output.shape[1]}")
    print(f"輸出特徵數: {output.shape[2]}")

    print(f"\n隱藏狀態形狀: {hidden.shape}")
    print(f"層數: {hidden.shape[0]}")
    print(f"批次大小: {hidden.shape[1]}")
    print(f"隱藏層單元數: {hidden.shape[2]}")

    prediction, hidden = model.predict_next(test_input)
    print(f"\n預測下一個值的形狀: {prediction.shape}")

    print("\n" + "=" * 50)
    print("模型測試成功！")
    print("=" * 50)