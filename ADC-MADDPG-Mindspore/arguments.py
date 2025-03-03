import time
import torch
import argparse

# device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
time_now = time.strftime('%y%m_%d%H%M')

def parse_args():
    parser = argparse.ArgumentParser("reinforcement learning experiments for multiagent environments")

    # environment
    parser.add_argument("--start_time", type=str, default=time_now, help="the time when start the game")
    parser.add_argument("--per_episode_max_len", type=int, default=400, help="maximum episode length")
    #parser.add_argument("--max_episode", type=int, default=18006, help="maximum episode length")
    parser.add_argument("--max_episode", type=int, default=16006, help="maximum episode length")
    parser.add_argument("--num-adversaries", type=int, default=1, help="number of adversaries")
    parser.add_argument("--per_obs_dim", type=int, default=15, help="dim of obs")
    parser.add_argument("--per_act_dim", type=int, default=2, help="dim of act")
    parser.add_argument("--jammer_act", action="store_true", default=True)

    # core training parameters
    parser.add_argument("--train_model", type=str, default='a', help="which model to be trained")
    # parser.add_argument("--device", default=device, help="torch device ")
    parser.add_argument("--learning_start_step", type=int, default=2000, help="learning start steps")
    #parser.add_argument("--max_grad_norm", type=float, default=0.5, help="max gradient norm for clip")
    parser.add_argument("--learning_fre", type=int, default=50, help="learning frequency")
    parser.add_argument("--tao", type=int, default=0.005, help="how depth we exchange the par of the nn")
    parser.add_argument("--lr_a", type=float, default=1e-4, help="learning rate for adam optimizer")
    parser.add_argument("--lr_c", type=float, default=1e-3, help="learning rate for adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--batch_size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--memory_size", type=int, default=1e6, help="number of data stored in the memory")
    parser.add_argument("--num_units_1", type=int, default=256, help="number of units in the mlp")
    parser.add_argument("--num_units_2", type=int, default=128, help="number of units in the mlp")
    parser.add_argument("--num_units_openai", type=int, default=256, help="number of units in the mlp")
    parser.add_argument("--num_attention_out", type=int, default=128, help="number of outputs of attention layer")

    # checkpointing
    parser.add_argument("--fre4save_model", type=int, default=10000, help="the number of the episode for saving the model")
    #parser.add_argument("--start_save_model", type=int, default=120000, help="the number of the episode for saving the model")
    parser.add_argument("--start_save_model", type=int, default=10000,
                        help="the number of the episode for saving the model")
    parser.add_argument("--save_dir", type=str, default="models", \
            help="directory in which training state and model should be saved")
    # parser.add_argument("--old_model_name", type=str, default="models/rpp_2402_021855_1502000/", \
    #         help="directory in which training state and model are loaded")
    parser.add_argument("--mix_model_name", type=str, default="models/attention_2408_240911_3002000/", \
            help="directory in which mix training state and model are loaded")
    parser.add_argument("--typical_model_name", type=str, default="models/typical_2403_091626_502000/", \
            help="directory in which typical training state and model are loaded")
    parser.add_argument("--jammer_model_name", type=str, default="models/jammer_2403_091000_702000/", \
            help="directory in which jammer training state and model are loaded")
    # evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", \
            help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", \
            help="directory where plot data is saved")
    return parser.parse_args()
