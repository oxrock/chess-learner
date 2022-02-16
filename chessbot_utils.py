from collections import deque
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow import keras, function
import numpy as np
import pickle


@function
def fast_predict(x, model) -> float:
    return model(x, training=False)

def create_piece_dict() -> dict:
    piece_encoder = {
        "p": [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "P": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "n": [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "N": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "b": [0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "B": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "r": [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        "R": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        "q": [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
        "Q": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "k": [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
        "K": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ".": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        "1": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "-1": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
    }
    return piece_encoder

def create_piece_decoder() -> dict:
    piece_encoder = {
        (-1, 0, 0, 0, 0, 0, 0, 0): "p",
        (1, 0, 0, 0, 0, 0, 0, 0): "P",
        (0, -1, 0, 0, 0, 0, 0, 0): "n",
        (0, 1, 0, 0, 0, 0, 0, 0): "N",
        (0, 0, -1, 0, 0, 0, 0, 0): "b",
        (0, 0, 1, 0, 0, 0, 0, 0): "B",
        (0, 0, 0, -1, 0, 0, 0, 0): "r",
        (0, 0, 0, 1, 0, 0, 0, 0): "R",
        (0, 0, 0, 0, -1, 0, 0, 0): "q",
        (0, 0, 0, 0, 1, 0, 0, 0): "Q",
        (0, 0, 0, 0, 0, -1, 0, 0): "k",
        (0, 0, 0, 0, 0, 1, 0, 0): "K",
        (0, 0, 0, 0, 0, 0, 1, 0): ".",
        (0, 0, 0, 0, 0, 0, 0, 1): "1",
        (0, 0, 0, 0, 0, 0, 0, -1): "-1",
    }
    return piece_encoder


def create_score_encoder() -> dict:
    piece_encoder = {
        (1, 0, 0, 0, 0, 0, 0, 0): 1,
        (-1, 0, 0, 0, 0, 0, 0, 0): -1,
        (0, 1, 0, 0, 0, 0, 0, 0): 3,
        (0, -1, 0, 0, 0, 0, 0, 0): -3,
        (0, 0, 1, 0, 0, 0, 0, 0): 3,
        (0, 0, -1, 0, 0, 0, 0, 0): -3,
        (0, 0, 0, 1, 0, 0, 0, 0): 5,
        (0, 0, 0, -1, 0, 0, 0, 0): -5,
        (0, 0, 0, 0, 1, 0, 0, 0): 9,
        (0, 0, 0, 0, -1, 0, 0, 0): -9,
        (0, 0, 0, 0, 0, 1, 0, 0): 1,
        (0, 0, 0, 0, 0, -1, 0, 0): -1,
        (0, 0, 0, 0, 0, 0, 1, 0): 0,
        (0, 0, 0, 0, 0, 0, 0, -1): 0,
        (0, 0, 0, 0, 0, 0, 0, 1): 0,
    }

    return piece_encoder

def load_lichess_game(game_path):
    game_data = None
    try:
        game_data = pickle.load(open(game_path, "rb"))
        os.remove(game_path)
        print("Loaded in lichess bot data!")
    except:
        pass #no games available to load

    return game_data

def flatten_board(encoded_board)->list:
    flat_board = []
    for x in encoded_board:
        #print(f"{x=}")
        for y in x:
            #print(f"{y=}")
            flat_board.append(y)
    return flat_board

def flatten_game(game_list)->list:
    game = []
    #print(game_list)
    for position in game_list:
        #print(position)
        game.append(flatten_board(position))
        #print("===============")
    return game

def create_training_data(mem, batchsize=64, randomize=False):
    if randomize:
        batch = random.sample(mem, batchsize)
    else:
        batch = list(mem)[len(mem)-batchsize:] + random.sample(mem, int(batchsize*0.1))

    random.shuffle(batch)
    #batch = list(mem)[len(mem) - batchsize:]
    #batch = [mem[i] for i in range(len(mem)-batchsize, len(mem))]

    features = np.array([x[0]for x in batch])
    labels = np.array([[x[1]] for x in batch])
    return features, labels



def load_memorybank():
    try:
        bank = pickle.load(open("gameplay_memory-v1.mem", "rb"))
        print("Loaded memory bank from file")
    except:
        print("Loading memorybank failed, creating new one.")
        bank = deque(maxlen=100_000)

    return bank


def load_trainer(encoder, env, handicap=5):
    latest_number = get_tracker_data("model_tracker.txt")[0]
    target_number = max(1, latest_number - handicap)
    return load_model(f"chess_model-trainer_{target_number}.mdl", encoder, env, verbose=False)


def load_model(model_path, encoder, env, verbose=True):
    while True:
        try:
            model = keras.models.load_model(model_path)
            if verbose:
                print(f"loaded {model_path} successfully!")
            return model

        except Exception as e:
            print(str(e))
            print(f"trying to load {model_path} again.")


def setup(encoder, env):
    feature_example = [encoder[x] for x in str(env.reset()).split()]
    feature_example.append([1])
    feature_example = flatten_board(feature_example)
    feature_example = np.array(feature_example)
    print(f"example shape is: {feature_example.shape}")
    model = keras.Sequential()
    model.add(
        keras.layers.Dense(
            feature_example.shape[0], input_shape=feature_example.shape
        )
    )

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation="relu"))
    model.add(keras.layers.Dense(512, activation="relu"))
    model.add(keras.layers.Dense(512, activation="relu"))
    model.add(keras.layers.Dense(512, activation="relu"))
    model.add(keras.layers.Dense(1, activation='linear'))
    # model.compile(optimizer="SGD", loss="mean_squared_error", metrics=["mean_absolute_error"])
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.1),
        loss="mean_squared_error",
        metrics=["mean_absolute_error"],
    )
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
        return None, None



def train_model(model, memory, experiences=0, epochs=10, bs=5000):
    print(f"Training on {int(experiences*1.1)} experiences in {bs} size batches over {epochs} epochs")
    for i in range(epochs):
        feat, labels = create_training_data(memory, batchsize=min(experiences, len(memory)))
        model.fit(feat, labels, epochs=1, batch_size=bs, verbose=1)


def get_tracker_data(tracker_path: str) -> int:
    with open(tracker_path, "r") as f:
        gen_count,training_count = f.read().split()
    return int(gen_count), int(training_count)


def set_tracker_data(tracker_path, gen_count: int, game_count: int):
    with open(tracker_path, "w+") as f:
        f.write(str(gen_count) + " " + str(game_count))


if __name__ == "__main__":
    pass
