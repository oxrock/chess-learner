import gym
import gym_chess
from players import Player, ML_Player, ML_Player_Trainer, ML_Distiller, chess_engine
from chessbot_utils import (
    create_piece_dict,
    load_memorybank,
    load_model,
    save_data,
    load_data,
    create_training_data,
    train_model,
    get_tracker_data,
    set_tracker_data,
    load_trainer,
    create_score_encoder,
    setup,
    load_lichess_game,
)
from multiprocessing import Process, Queue
#import cProfile
#import pstats
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
stockfish_path = "engines\stockfish_13\stockfish_13_win_x64_bmi2.exe"
#print(stockfish_path)
maia_path = "engines\maia\lc0.exe"
acqua_path = "engines\acqua\acqua.exe"
chosen_engine = stockfish_path
#chosen_engine = stockfish_path
#chosen_engine = acqua_path
chosen_name = "stockfish"
#chosen_name = "acqua"
#chosen_name = "maia"

epsilon = 0.95
search_depth = 2
train_req = 500
trainer_save_req = 2000
epochs_targ = 1


def classless_trainer(
    q: Queue, training_method: int, rewards_boosting: bool, verbose: bool
):  # 0: random mover opponent,1:random mirrors, 2:old version opponent, 3:mirror ML match, 4:distiller vs trainer, 5: ML player vs engine, 6: random vs engine, 7: option 2 with automatic distilling, 9: trainer vs engine
    env = gym.make("Chess-v0")
    piece_encoder = create_piece_dict()
    global train_req
    if training_method not in [1, 6]:
        model = load_model("chess_model-v1.mdl", piece_encoder, env)
        old_model = load_trainer(piece_encoder, env)
    else:
        model = None
        old_model = None
    update_req = 3

    if training_method in [1, 6]:
        update_req *= 100_000
    # if training_method == 4:
    #     update_req = int(train_req/(search_depth*8))
    distil_req = 10
    score_encoder = create_score_encoder()

    def generate_players(game_count: int) -> (Player, Player):
        if training_method == 0:
            if game_count % 2 == 0:
                white_player = Player(1, piece_encoder, score_encoder, rewards_boosting)
                black_player = ML_Player(
                    0, piece_encoder, score_encoder, rewards_boosting, model
                )
            else:
                white_player = ML_Player(
                    1, piece_encoder, score_encoder, rewards_boosting, model
                )
                black_player = Player(0, piece_encoder, score_encoder, rewards_boosting)

        elif training_method == 1:
            white_player = Player(1, piece_encoder, score_encoder, rewards_boosting)
            black_player = Player(0, piece_encoder, score_encoder, rewards_boosting)

        elif training_method == 2:
            if game_count % 2 == 0:
                white_player = ML_Player(
                    1,
                    piece_encoder,
                    score_encoder,
                    rewards_boosting,
                    model,
                    epsilon=epsilon,
                )
                black_player = ML_Player_Trainer(
                    0, piece_encoder, score_encoder, rewards_boosting, old_model
                )
            else:
                white_player = ML_Player_Trainer(
                    1, piece_encoder, score_encoder, rewards_boosting, old_model
                )
                black_player = ML_Player(
                    0,
                    piece_encoder,
                    score_encoder,
                    rewards_boosting,
                    model,
                    epsilon=epsilon,
                )

        elif training_method == 3:
            white_player = ML_Player(
                1, piece_encoder, score_encoder, rewards_boosting, model
            )
            black_player = ML_Player(
                0, piece_encoder, score_encoder, rewards_boosting, model
            )

        elif training_method == 4:
            if game_count % 2 == 0:
                white_player = ML_Distiller(
                    1,
                    piece_encoder,
                    score_encoder,
                    rewards_boosting,
                    model,
                    depth=search_depth,
                )
                # black_player = ML_Player(
                #     0,
                #     piece_encoder,
                #     score_encoder,
                #     rewards_boosting,
                #     old_model,
                #     epsilon=epsilon,
                # )
                black_player = ML_Player_Trainer(
                    0, piece_encoder, score_encoder, rewards_boosting, old_model
                )
                #black_player = Player(0, piece_encoder, score_encoder, rewards_boosting)
            else:
                # white_player = ML_Player(
                #     1,
                #     piece_encoder,
                #     score_encoder,
                #     rewards_boosting,
                #     old_model,
                #     epsilon=epsilon,
                # )
                white_player = ML_Player_Trainer(
                    0, piece_encoder, score_encoder, rewards_boosting, old_model
                )
                #white_player = Player(1, piece_encoder, score_encoder, rewards_boosting)
                black_player = ML_Distiller(
                    0,
                    piece_encoder,
                    score_encoder,
                    rewards_boosting,
                    model,
                    depth=search_depth,
                )

        elif training_method == 5:
            if game_count % 2 == 0:
                white_player = chess_engine(
                    1,
                    piece_encoder,
                    score_encoder,
                    rewards_boosting,
                    chosen_engine,
                    0.05,
                    chosen_name,
                )
                black_player = ML_Player(
                    0, piece_encoder, score_encoder, rewards_boosting, model
                )
                # black_player = ML_Distiller(
                #     0,
                #     piece_encoder,
                #     score_encoder,
                #     rewards_boosting,
                #     model,
                #     depth=search_depth,
                # )
            else:
                white_player = ML_Player(
                    1, piece_encoder, score_encoder, rewards_boosting, model
                )
                # white_player = ML_Distiller(
                #     1,
                #     piece_encoder,
                #     score_encoder,
                #     rewards_boosting,
                #     model,
                #     depth=search_depth,
                # )
                black_player = chess_engine(
                    0,
                    piece_encoder,
                    score_encoder,
                    rewards_boosting,
                    chosen_engine,
                    0.05,
                    chosen_name,
                )

        elif training_method == 6:
            if game_count % 2 == 0:
                white_player = chess_engine(
                    1,
                    piece_encoder,
                    score_encoder,
                    rewards_boosting,
                    chosen_engine,
                    0.05,
                    chosen_name,
                )
                black_player = Player(0, piece_encoder, score_encoder, rewards_boosting)
                # black_player = ML_Distiller(0, piece_encoder, rewards_boosting, model,depth=search_depth)
            else:
                white_player = Player(1, piece_encoder, score_encoder, rewards_boosting)
                # white_player = ML_Distiller(1, piece_encoder, rewards_boosting, model,depth=search_depth)
                black_player = chess_engine(
                    0,
                    piece_encoder,
                    score_encoder,
                    rewards_boosting,
                    chosen_engine,
                    0.05,
                    chosen_name,
                )

        elif training_method == 7:
            if game_count < distil_req:
                if game_count % 2 == 0:
                    white_player = ML_Distiller(
                        1,
                        piece_encoder,
                        score_encoder,
                        rewards_boosting,
                        model,
                        depth=search_depth,
                    )
                    black_player = ML_Player(
                        0,
                        piece_encoder,
                        score_encoder,
                        rewards_boosting,
                        old_model,
                        epsilon=epsilon,
                    )
                else:
                    white_player = ML_Player(
                        1,
                        piece_encoder,
                        score_encoder,
                        rewards_boosting,
                        old_model,
                        epsilon=epsilon,
                    )
                    black_player = ML_Distiller(
                        0,
                        piece_encoder,
                        score_encoder,
                        rewards_boosting,
                        model,
                        depth=search_depth,
                    )
            else:
                if game_count % 2 == 0:
                    # white_player = ML_Distiller(1, piece_encoder, rewards_boosting, model,epsilon=epsilon, depth=search_depth)
                    white_player = ML_Player(
                        1,
                        piece_encoder,
                        score_encoder,
                        rewards_boosting,
                        model,
                        epsilon=epsilon,
                    )
                    black_player = ML_Player_Trainer(
                        0, piece_encoder, score_encoder, rewards_boosting, old_model
                    )
                else:
                    white_player = ML_Player_Trainer(
                        1, piece_encoder, score_encoder, rewards_boosting, old_model
                    )
                    # black_player = ML_Distiller(0, piece_encoder, rewards_boosting, model,epsilon=epsilon, depth=search_depth)
                    black_player = ML_Player(
                        0,
                        piece_encoder,
                        score_encoder,
                        rewards_boosting,
                        model,
                        epsilon=epsilon,
                    )

        elif training_method == 8:  # trainer 45 last model before this change
            if game_count < distil_req:
                if game_count % 2 == 0:
                    white_player = chess_engine(
                        1,
                        piece_encoder,
                        score_encoder,
                        rewards_boosting,
                        chosen_engine,
                        0.05,
                        chosen_name,
                    )
                    black_player = ML_Player(
                        0,
                        piece_encoder,
                        score_encoder,
                        rewards_boosting,
                        old_model,
                        epsilon=epsilon,
                    )
                else:
                    white_player = ML_Player(
                        1,
                        piece_encoder,
                        score_encoder,
                        rewards_boosting,
                        old_model,
                        epsilon=epsilon,
                    )
                    black_player = chess_engine(
                        0,
                        piece_encoder,
                        score_encoder,
                        rewards_boosting,
                        chosen_engine,
                        0.05,
                        chosen_name,
                    )
            else:
                if game_count % 2 == 0:
                    # white_player = ML_Distiller(1, piece_encoder, rewards_boosting, model,epsilon=epsilon, depth=search_depth)
                    white_player = ML_Player(
                        1,
                        piece_encoder,
                        score_encoder,
                        rewards_boosting,
                        model,
                        epsilon=epsilon,
                    )
                    black_player = ML_Player_Trainer(
                        0, piece_encoder, score_encoder, rewards_boosting, old_model
                    )
                else:
                    white_player = ML_Player_Trainer(
                        1, piece_encoder, score_encoder, rewards_boosting, old_model
                    )
                    # black_player = ML_Distiller(0, piece_encoder, rewards_boosting, model,epsilon=epsilon, depth=search_depth)
                    black_player = ML_Player(
                        0,
                        piece_encoder,
                        score_encoder,
                        rewards_boosting,
                        model,
                        epsilon=epsilon,
                    )

        elif training_method == 9: #last trainer before implementation: 16
            if game_count % 2 == 0:
                white_player = chess_engine(
                    1,
                    piece_encoder,
                    score_encoder,
                    rewards_boosting,
                    chosen_engine,
                    0.05,
                    chosen_name,
                )
                black_player = ML_Player_Trainer(
                    0, piece_encoder, score_encoder, rewards_boosting, old_model
                )
            else:
                white_player = ML_Player_Trainer(
                    1, piece_encoder, score_encoder, rewards_boosting, old_model
                )
                black_player = chess_engine(
                    0,
                    piece_encoder,
                    score_encoder,
                    rewards_boosting,
                    chosen_engine,
                    0.05,
                    chosen_name,
                )

        else:
            print(f"training method {training_method} not recognized!")
            print("spawning random movers")
            white_player = Player(1, piece_encoder, score_encoder, rewards_boosting)
            black_player = Player(0, piece_encoder, score_encoder, rewards_boosting)

        return white_player, black_player

    def play_match(player_white, player_black) -> (Player, Player, str):
        #profiler = cProfile.Profile()
        #profiler.enable()
        state = env.reset()
        done = False
        turn = state.turn
        observation = state
        start = time.time()
        turns = 0
        while not done:
            if turn:
                action = player_white.generate_move(observation.copy(stack=False))
            else:
                action = player_black.generate_move(observation.copy(stack=False))

            observation, reward, done, _ = env.step(action)
            turn = observation.turn
            turns += 1

        # print(f"A game with {turns} turns took {time.time()-start} seconds.")
        player_white.quit()
        player_black.quit()
        #profiler.disable()
        #stats = pstats.Stats(profiler).sort_stats('cumtime')
        #stats.print_stats()
        return (player_white, player_black, observation.result())

    def train(updates=False):
        game_count = 0
        recorded_draws = 0
        update_count = 0
        while True:
            white_player, black_player = generate_players(game_count)
            _w, _b, result = play_match(white_player, black_player)

            if result == "1-0":
                if verbose:
                    print(f"{str(white_player)} victory over {str(black_player)}")
                white_data = white_player.finalize_data(1)
                black_data = black_player.finalize_data(-1)

            elif result == "0-1":
                if verbose:
                    print(f"{str(black_player)} victory over {str(white_player)}")
                white_data = white_player.finalize_data(-1)
                black_data = black_player.finalize_data(1)

            else:
                #if verbose:
                    #print(f"{str(black_player)} draws with {str(white_player)}")
                white_data = white_player.finalize_data(-0.5, draw=True)
                black_data = black_player.finalize_data(-0.5, draw=True)

            training_data = white_data + black_data + [result]

            if len(training_data) > 0:
                q.put(training_data)
                update_count += 1

            if updates and update_count >= update_req:
                return None

            game_count += 1

    old_count = 0
    while True:
        updates = True if training_method not in [1, 6] else False
        train(updates=updates)
        model = load_model("chess_model-v1.mdl", piece_encoder, env, verbose=False)
        old_count += 1
        if old_count % 5 == 0:
            old_model = load_trainer(piece_encoder, env)


def data_gobbler(q: Queue, training_method: int):
    env = gym.make("Chess-v0")
    encoder = create_piece_dict()
    model, memory_bank = load_data("chess_model-v1.mdl", encoder, env)
    train_count = 0
    experience_count = 0
    draw_count = 0
    start_time = time.time()
    model_count,game_count = get_tracker_data("model_tracker.txt")
    _train_req = train_req
    _trainer_save_req = trainer_save_req
    if training_method in [1, 6]:
        _train_req *= 20
        _trainer_save_req *= 20

    while True:
        if not q.empty():
            data = q.get()
            result = ""
            if type(data[-1]) is str:
                result = data.pop()
            game_count += 1
            train_count += 1
            for item in data:
                experience_count += 1
                memory_bank.append(item)
                # print(item)
            if result != "0-1" and result != "1-0":
                draw_count += 1

            #print(f"Games: {game_count} | Draws: {draw_count}")

            if train_count >= _train_req:
                #epochs_targ = 1
                train_model(
                    model,
                    memory_bank,
                    experiences=experience_count,
                    epochs=20,
                    bs=min(100_000, int(experience_count*1.1))
                )
                save_data("chess_model-v1.mdl", model, memory_bank)
                set_tracker_data("model_tracker.txt", model_count, game_count)
                print(f"{game_count} matches trained so far!")
                print(f"{draw_count}/{_train_req} games were draws in this chunk")
                t = time.time()
                minutes = (t - start_time) / 60
                print(f"Averaging {_train_req/minutes} games a minute")
                train_count = 0
                experience_count = 0
                draw_count = 0
                print("++++++++++++++++++++++")
                start_time = t

            if game_count % _trainer_save_req == 0:
                model_count += 1
                model.save(f"chess_model-trainer_{model_count}.mdl")
                set_tracker_data("model_tracker.txt", model_count,game_count)
        else:
            bot_data = load_lichess_game("D:/coding projects/chess_stuffz/lichess-bot-rework/game_data.pkl")
            if bot_data != None:
                q.put(bot_data)


if __name__ == "__main__":
    # env = gym.make("Chess-v0")
    # encoder = create_piece_dict()
    # #decoder = create_piece_decoder()
    # score_encoder = create_score_encoder()
    # setup(encoder, env)
    # model, memory_bank = load_data("chess_model-v1.mdl", encoder, env)
    # train_model(model, memory_bank, experiences=len(memory_bank), epochs=50, bs=len(memory_bank))
    # save_data("chess_model-v1.mdl", model, memory_bank)
    # print("done")
    # feats,labels = create_training_data(memory_bank)
    # print(feats.shape)

    try:
        my_q = Queue()
        set_up = input(
            "Perform first time setup? (May error if requisite files are missing) Y/N\n"
        )
        if set_up.lower() == "y":
            setup(create_piece_dict(), gym.make("Chess-v0"))
            set_tracker_data("model_tracker.txt", 1, 0)
        # 0: random mover opponent,1:random mirrors, 2:old version opponent, 3:mirror ML match, 4:distiller vs trainer, 5: ML player vs engine
        selected_method = int(
            input(
                "========================================\nChoose training method - 0:ML agent vs random mover, 1:random mirror matches, 2:ML agent vs older version, 3: mirror ML agents, 4: distiller vs trainer, 5: ML agent vs chess engine, 6: random vs engine, 7: option 2 with automatic distilling, 8: Same as 7 but with chess engine as distiller agent, 9: Engine vs ML Trainer\n"
            )
        )
        boosted_rewards = input("Would you like to artificially boost rewards?: \n").lower()
        num_processes = int(input("How many processes would you like to run?: \n"))
        verbosity = input("Verbose mode enabled? Y/N: \n")
        print(
            f"Running training method {selected_method} with {num_processes} processes"
        )
        print(
            "+++++++++++++++++++++++++++++++++\nPress control+c in this console to end training session\n+++++++++++++++++++++++++++++++++"
        )
        if selected_method not in [
            1,
            6,
        ]:  # gpu training has issues when run in more than 1 process so only enabled for non ML agent matches
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        fo_pool = [
            Process(
                target=classless_trainer,
                args=(
                    my_q,
                    selected_method,
                    True if boosted_rewards == "y" else False,
                    True if verbosity.lower() == "y" else False,
                ),
                daemon=True,
            )
            for i in range(num_processes)
        ]
        for p in fo_pool:
            p.start()
        data_gobbler(my_q, selected_method)
        for p in fo_pool:
            p.terminate()

    except KeyboardInterrupt:
        for p in fo_pool:
            p.terminate()
        input("Processes terminated. Press ENTER to exit.")
