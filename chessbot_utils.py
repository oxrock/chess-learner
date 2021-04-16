from collections import deque
import random
from tensorflow import keras,function
import numpy as np
import pickle


@function
def fast_predict(x,model) -> float:
    return float(model(x, training=False)[0][0])


# def create_piece_dict() -> dict:
#     piece_encoder = {
#         "p": 1 / 6,
#         "P": -(1 / 6),
#         "n": 2 / 6,
#         "N": -(2 / 6),
#         "b": 3 / 6,
#         "B": -(3 / 6),
#         "r": 4 / 6,
#         "R": -(4 / 6),
#         "q": 5 / 6,
#         "Q": -(5 / 6),
#         "k": 1,
#         "K": -1,
#         ".": 0,
#     }
#     return piece_encoder
def create_piece_dict() -> dict:
    piece_encoder = {
        "p": [1, 0, 0, 0, 0, 0, 0],
        "P": [-1, 0, 0, 0, 0, 0, 0],
        "n": [0, 1, 0, 0, 0, 0, 0],
        "N": [0, -1, 0, 0, 0, 0, 0],
        "b": [0, 0, 1, 0, 0, 0, 0],
        "B": [0, 0, -1, 0, 0, 0, 0],
        "r": [0, 0, 0, 1, 0, 0, 0],
        "R": [0, 0, 0, -1, 0, 0, 0],
        "q": [0, 0, 0, 0, 1, 0, 0],
        "Q": [0, 0, 0, 0, -1, 0, 0],
        "k": [0, 0, 0, 0, 0, 1, 0],
        "K": [0, 0, 0, 0, 0, -1, 0],
        ".": [0, 0, 0, 0, 0, 0, 1]
    }
    return piece_encoder

def create_score_encoder() -> dict:
    piece_encoder = {
        (1, 0, 0, 0, 0, 0, 0):1,
        (-1, 0, 0, 0, 0, 0, 0):-1,
        (0, 1, 0, 0, 0, 0, 0): 2,
        (0, -1, 0, 0, 0, 0, 0): -2,
        (0, 0, 1, 0, 0, 0, 0): 3,
        (0, 0, -1, 0, 0, 0, 0): -3,
        (0, 0, 0, 1, 0, 0, 0): 5,
        (0, 0, 0, -1, 0, 0, 0): -5,
        (0, 0, 0, 0, 1, 0, 0): 13,
        (0, 0, 0, 0, -1, 0, 0): -13,
        (0, 0, 0, 0, 0, 1, 0): 100,
        (0, 0, 0, 0, 0, -1, 0): -100,
        (0, 0, 0, 0, 0, 0, 1): 0
    }

    return piece_encoder



def load_memorybank():
    try:
        bank = pickle.load(open("gameplay_memory-v1.mem", "rb"))
        print("Loaded memory bank from file")
    except:
        print("Loading memorybank failed, creating new one.")
        bank = deque(maxlen=200000)

    return bank

def load_trainer(encoder,env,handicap=4):
    latest_number = get_tracker_data("model_tracker.txt")
    target_number = max(1,latest_number-handicap)
    return load_model(f"chess_model-trainer_{target_number}.mdl", encoder, env)

# def load_model(model_path, encoder, env):
#     while True:
#         try:
#             model = keras.models.load_model(model_path)
#             print(f"loaded {model_path} successfully!")
#             return model
#
#         except Exception as e:
#             print(str(e))
#             print(f"trying to load {model_path} again.")
#         # feature_example = [encoder[x] for x in str(env.reset()).split()]
#         # feature_example.append(1)  # 1 for white, -1 for black
#         # feature_example = np.array([feature_example])
#         # print("No model information found, creating new model!")
#         # model = keras.Sequential()
#         # model.add(
#         #     keras.layers.Dense(
#         #         len(feature_example), input_shape=feature_example.shape[1:]
#         #     )
#         # )
#         #
#         # model.add(keras.layers.Dense(1024, activation="relu"))
#         # model.add(keras.layers.Dense(1024, activation="relu"))
#         # model.add(keras.layers.Dense(1024, activation="relu"))
#         # model.add(keras.layers.Dense(1, activation="linear"))
#         # #model.compile(optimizer="SGD", loss="mean_squared_error", metrics=["mean_absolute_error"])
#         # model.compile(optimizer="adam", loss="mean_squared_error")
#         # model.save(model_path)
#         # # model.save("chess_model-v1.mdl")
#
#     return model
def load_model(model_path, encoder, env):
    while True:
        try:
            model = keras.models.load_model(model_path)
            print(f"loaded {model_path} successfully!")
            return model

        except Exception as e:
            print(str(e))
            print(f"trying to load {model_path} again.")

def setup(encoder,env):
    feature_example = [encoder[x] for x in str(env.reset()).split()]
    feature_example.append([1, 0, 0, 0, 0, 0, 0])
    feature_example = np.array(feature_example)

    print("No model information found, creating new model!")
    model = keras.Sequential()
    model.add(
        keras.layers.Dense(
            len(feature_example), input_shape=feature_example.shape[1:],
            kernel_initializer='normal'
        )
    )

    model.add(
        keras.layers.Dense(1024, kernel_initializer='normal', activation="tanh", ))
    model.add(
        keras.layers.Dense(1024, kernel_initializer='normal', activation="tanh"))
    model.add(keras.layers.Dense(1, kernel_initializer='normal'))
    # model.compile(optimizer="SGD", loss="mean_squared_error", metrics=["mean_absolute_error"])
    model.compile(optimizer=keras.optimizers.SGD(lr=0.001), loss="mean_squared_error")
    model.save("chess_model-v1.mdl")
    model.save("chess_model-trainer_1.mdl")


def save_data(model_path, model, memory):
    try:
        # model.save("chess_model-v1.mdl")
        model.save(model_path)
        pickle.dump(memory, open("gameplay_memory-v1.mem", "wb"))
        print("Data saved.")
    except Exception as e:
        print("Failed to save data!")
        print(str(e))


def load_data(model_path, encoder, env):
    try:
        return load_model(model_path, encoder, env), load_memorybank()
    except:
        print("Loading data failed!")
        return None,None

def create_training_data(mem, batchsize=64,randomize=False):
    if randomize:
        batch = random.sample(mem, batchsize)
    else:
        batch = list(mem)[-batchsize:] +random.sample(mem, int(batchsize/10))
    features = np.array([x[0] for x in batch])
    labels = np.array([x[1] for x in batch])
    return features, labels


def train_model(model, memory, batch_size=1024, epochs=1):
    batch_size = min(batch_size,len(memory))
    feat, labels = create_training_data(memory, batchsize=batch_size)
    model.fit(feat, labels, epochs=epochs, batch_size=32, verbose=1)

def get_tracker_data(tracker_path:str) -> int:
    with open(tracker_path,'r') as f:
        count = int(f.read())
    return count

def set_tracker_data(tracker_path, number:int):
    with open(tracker_path, 'w+') as f:
        f.write(str(number))

if __name__ == "__main__":
    pass

