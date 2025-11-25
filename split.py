from sklearn.model_selection import train_test_split
import numpy as np


allData=np.load("jammingData/data_3channel.npy")
allLabels_int=np.load("jammingData/labels_3channel.npy")
X_train, X_test, y_train, y_test = train_test_split(
        allData, allLabels_int, test_size=0.3, random_state=42, stratify=allLabels_int
    )
np.save(f"jammingData/X_3channels_train.npy", X_train)
np.save(f"jammingData/y_channels_train.npy", y_train)
np.save(f"jammingData/X_3channels_test.npy", X_test)
np.save(f"jammingData/y_channels_test.npy", y_test)