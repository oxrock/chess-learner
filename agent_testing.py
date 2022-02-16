import gym
import gym_chess
from players import *
from chessbot_utils import *
from time import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def load_all_models(encoder, env):
    return [
        load_model(f"chess_model-trainer_{x+1}.mdl", encoder, env)
        for x in range(get_tracker_data("model_tracker.txt")[0])
    ]


def play_match(player_white, player_black, env) -> (Player, Player, str):
    state = env.reset()
    done = False
    turn = state.turn
    observation = state
    while not done:
        if turn:
            action = player_white.generate_move(observation.copy(stack=False))
        else:
            action = player_black.generate_move(observation.copy(stack=False))

        observation, reward, done, _ = env.step(action)
        turn = observation.turn

    player_white.quit()
    player_black.quit()
    return (player_white, player_black, observation.result())


def testing_gauntlet():
    env = gym.make("Chess-v0")
    piece_encoder = create_piece_dict()
    score_encoder = create_score_encoder()
    models = load_all_models(piece_encoder, env)
    model_tallies = {x + 1: 0 for x in range(len(models))}
    match_count = 0
    start_time = time()
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            for k in range(2):
                players = [i + 1, j + 1]
                if k % 2 == 0:
                    player_white = ML_Player_Trainer(
                        1, piece_encoder, score_encoder, False, models[i]
                    )
                    player_black = ML_Player_Trainer(
                        0, piece_encoder, score_encoder, False, models[j]
                    )
                else:
                    players = [j + 1, i + 1]
                    player_white = ML_Player_Trainer(
                        1, piece_encoder, score_encoder, False, models[j], epsilon=1
                    )
                    player_black = ML_Player_Trainer(
                        0, piece_encoder, score_encoder, False, models[i], epsilon=1
                    )

                white, black, result = play_match(player_white, player_black, env)
                white.quit()
                black.quit()
                if result == "1-0":
                    model_tallies[players[0]] += 1
                elif result == "0-1":
                    model_tallies[players[1]] += 1
                else:
                    model_tallies[players[0]] += 0.5
                    model_tallies[players[1]] += 0.5
                match_count += 1

        print(
            f"model {i+1} has finished their games with a score of {model_tallies[i+1]}"
        )

    print(f"Running {match_count} matches took {time()-start_time} seconds")
    print(model_tallies)


if __name__ == "__main__":
    # gpu Running all those matches took 133.87403535842896 seconds
    # cpu Running all those matches took 100.78716683387756 seconds
    testing_gauntlet()
