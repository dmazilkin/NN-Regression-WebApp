from keras.api import Sequential
from keras.api.layers import Dense, Input
from keras.api.callbacks import EarlyStopping
from keras.api.optimizers import SGD
from keras.api.initializers import RandomNormal
from keras.api.metrics import R2Score

from helpers.preprocess_data import preprocess_data
from helpers.generate_data import create_dataset
from helpers.config import TRAINED_MODEL

DATASET_SIZE = 5000

# Generate dataset for training
print('CREATING DATASET...', end=' ')
df = create_dataset(DATASET_SIZE)
print('[DONE]')
print('PREPROCESS DATA...', end=' ')
X_train, Y_train = preprocess_data(df)
print('[DONE]')
# Setup neural network
nn = Sequential()
nn.add(Input(shape=(X_train.shape[1],), ))
nn.add(Dense(units=32, activation='sigmoid', kernel_initializer=RandomNormal(0, 1)))
nn.add(Dense(units=1, activation='linear', kernel_initializer=RandomNormal(0, 1)))
sgd_optimizer = SGD(learning_rate=0.1)
print('TRAINING MODEL...', end=' ')
nn.compile(optimizer=sgd_optimizer, loss='mse', metrics=['mse', 'mae', R2Score()])
print('[DONE]')
early_stop = EarlyStopping(monitor='loss', min_delta=0.0001, patience=15, restore_best_weights=True)
nn.fit(X_train, Y_train, epochs=300, batch_size=32, shuffle=True, callbacks=[early_stop], verbose=0)
# Save neural network parameters
print('SAVING MODEL...', end=' ')
nn.save(TRAINED_MODEL)
print('[DONE]')