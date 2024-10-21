from model import (
    DebugModel,
    RNNModel,
    LSTMModel,
    GRUModel,
    xLSTM,
    BaseTransformerModel,
    Crossformer,
    SwitchTransformer
)

from torch.nn import MSELoss

MODEL_LIST = {
    "DebugModel":DebugModel,
    "RNN": RNNModel,
    "LSTM":LSTMModel,
    "GRU": GRUModel,
    "xLSTM":xLSTM,
    "BaseTransformer": BaseTransformerModel,
    "Crossformer": Crossformer,
    "Switch_Former": SwitchTransformer
}
LOSS_LIST = {
    "MSELoss":MSELoss
}