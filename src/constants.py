# Column groups (decided in the EDA notebook)

# Identifiers: dropped before modelling
ID_COLS = ["UDI", "Product ID"]

# Labels: used ONLY for final evaluation, never for training or tuning
TARGET_COL = "Machine failure"
LABEL_COLS = ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]

# Categorical column: dropped (no effect on sensor behaviour, and it distorted IForest)
CATEGORICAL_COL = "Type"

# Raw sensor columns
SENSOR_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Engineered features
DERIVED_COLS = ["temp_diff", "power_W"]

# Preprocessing groups
SKEWED_COLS = ["Rotational speed [rpm]", "power_W"]          # need Yeo-Johnson
NORMAL_COLS = [                                                # StandardScaler only
    "Air temperature [K]",
    "Process temperature [K]",
    "Torque [Nm]",
    "Tool wear [min]",
    "temp_diff",
]

# Final model input, in a fixed order
FEATURE_COLS = SKEWED_COLS + NORMAL_COLS

# Columns an uploaded CSV must contain (used for validation in the API)
REQUIRED_COLS = SENSOR_COLS

# Columns the training file must contain (sensors + labels)
TRAIN_REQUIRED_COLS = SENSOR_COLS + LABEL_COLS