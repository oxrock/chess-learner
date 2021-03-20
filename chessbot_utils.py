from collections import deque
import random
from tensorflow import keras
import numpy as np
import pickle

def create_piece_dict() -> dict:
    piece_encoder = {
        "p": 1 / 6,
        "P": -(1 / 6),
        "n": 2 / 6,
        "N": -(2 / 6),
        "b": 3 / 6,
        "B": -(3 / 6),
        "r": 4 / 6,
        "R": -(4 / 6),
        "q": 5 / 6,
        "Q": -(5 / 6),
        "k": 1,
        "K": -1,
        ".": 0,
    }
    return piece_encoder

def load_memorybank():
    try:
        bank = pickle.load( open( "gameplay_memory-v1.mem", "rb" ) )
        print("Loaded memory bank from file")
    except:
        print("Loading memorybank failed, creating new one.")
        bank = deque(maxlen=50000)

    return bank

def load_model(model_path,encoder,env):
    feature_example = [encoder[x] for x in str(env.reset()).split()]
    feature_example.append(1) #1 for white, -1 for black
    feature_example = np.array([feature_example])

    try:
        model = keras.models.load_model(model_path)
        print("loaded model successfully!")

    except:
        print("No model information found, creating new model!")
        model = keras.Sequential()
        model.add(keras.layers.Dense(len(feature_example), input_shape=feature_example.shape[1:]))
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dense(1, activation='relu'))
        model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='mean_squared_error', metrics=["accuracy"])
        model.save(model_path)
        #model.save("chess_model-v1.mdl")

    return model

def save_data(model_path,model,memory):
    try:
        #model.save("chess_model-v1.mdl")
        model.save(model_path)
        pickle.dump(memory, open("gameplay_memory-v1.mem", "wb"))
        print("Data saved.")
    except Exception as e:
        print("Failed to save data!")
        print(str(e))

def load_data(model_path,encoder,env):
    return load_model(model_path,encoder,env),load_memorybank()

def create_training_data(mem,batchsize=64):
    batch = random.sample(mem,batchsize)
    features = np.array([x[0] for x in batch])
    labels = np.array([[[x[1] for x in batch]]])
    return features,labels


def train_model(model,memory):
    if len(memory) > 1000:
        feat, labels = create_training_data(memory, batchsize=128)
        model.fit(feat[0], labels, epochs=1, batch_size=32, verbose=1)

    else:
        print(f"Only {len(memory)} experiences in the memorybank so far. Waiting for 1000 minimum")

if __name__ == "__main__":
    pass