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
    GaussianLoss_Module
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
    "GaussianLoss":GaussianLoss_Module
}

SCALE_COLS = {
    'Age': {'mean': 61.29849357562402, 'std': 18.19418807291505},
    'Weight': {'mean': 171.02129628381118, 'std': 35.70613888528012}, 
    'Pulse': {'mean': 77.08666582476043, 'std': 7.968495718027011}, 
    'Systolic': {'mean': 127.54706114658761, 'std': 10.966879454975894}, 
    'Diastolic': {'mean': 76.12535998482291, 'std': 7.157650023686608},
    'Temperature': {'mean': 97.37746139507836, 'std': 0.4029931929148571}, 
    'BMI': {'mean': 28.31795665485163, 'std': 4.738855199226916}, 
    'RespiratoryRate': {'mean': 15.769906806459147, 'std': 0.8270299872958025}, 
    'OxygenSaturation': {'mean': 97.54472085087278, 'std': 0.7741982235415734}, 
    'OxygenConcentration': {'mean': 21.0, 'std': 0.0}, 
    'Heights': {'mean': 65.38816933619539, 'std': 3.761805338639916}, 
    'CancelledRateThePast6Months': {'mean': 0.14599177522116094, 'std': 0.13431310129158702}, 
    'RescheduledRateThePast6Months': {'mean': 0.15612077044526768, 'std': 0.1339384312562343}, 
    'CancelledAppointmentsSinceLastEncounter': {'mean': 0.2672286817380437, 'std': 0.6500633444773264}, 
    'RescheduledAppointmentsSinceLastEncounter': {'mean': 0.30636702448168895, 'std': 0.7095928735594833}, 
    'Current ICD Count': {'mean': 13.32399741123435, 'std': 10.304246862455331}, 
    '6months ICD Count': {'mean': 21.18070030574215, 'std': 11.426550590906622}, 
    'allergies_count': {'mean': 0.39879265326162155, 'std': 0.9348012162877456}, 
    'vaccination_count': {'mean': 0.24408043027070456, 'std': 0.6698808172313843}, 
    'Average Visit Pattern': {'mean': 36.18208549930163, 'std': 38.270728456473265},
    'Target': {'mean': 27.660040393670915, 'std': 36.42564782642134}, 
}
