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

from loss import (
    MSE_Module,
    GaussianLoss_Module,
    StdReg_Module,
    LogCoshLoss
)

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
    "MSELoss":MSE_Module,
    "GaussianLoss":GaussianLoss_Module,
    "StdReg":StdReg_Module,
    "LogCosh":LogCoshLoss
}


