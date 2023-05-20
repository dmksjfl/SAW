import numpy as np
import torch
import gym
import argparse
import os
import d4rl
from tqdm import trange
from coolname import generate_slug
import time
import json
from log import Logger

import utils
from utils import VideoRecorder
import SAW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment


def eval_policy(args, iter, video: VideoRecorder, logger: Logger, 
                policy, env_name, seed, min_state, max_state, seed_offset=100, 
                eval_episodes=10, offline=True):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + seed_offset)

    lengths = []
    returns = []
    avg_reward = 0.
    for _ in range(eval_episodes):
        # video.init(enabled=(args.save_video and _ == 0))
        state, done = eval_env.reset(), False
        # video.record(eval_env)
        steps = 0
        episode_return = 0
        while not done:
            state = 2 * (np.array(state).reshape(1, -1) - min_state)/(max_state - min_state) - 1
            action = policy.select_action(state)
            state, reward, done, _ = eval_env.step(action)
            # video.record(eval_env)
            avg_reward += reward
            episode_return += reward
            steps += 1
        lengths.append(steps)
        returns.append(episode_return)
        # video.save(f'eval_s{iter}_r{str(episode_return)}.mp4')

    avg_reward /= eval_episodes

    if offline:
        d4rl_score = eval_env.get_normalized_score(avg_reward) * 100

        print("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes: {d4rl_score:.3f}")
        print("---------------------------------------")
        logger.log('eval/offline lengths_mean', np.mean(lengths), iter)
        logger.log('eval/offline returns_mean', np.mean(returns), iter)
        logger.log('eval/offline d4rl_score', d4rl_score, iter)

        return d4rl_score
    else:
        print("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
        print("---------------------------------------")

        logger.log('eval/online lengths_mean', np.mean(lengths), iter)
        logger.log('eval/online returns_mean', np.mean(returns), iter)
        return avg_reward


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--policy", default="SAW")                 # Policy name
    parser.add_argument("--env", default="hopper-medium-v2")        # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
    parser.add_argument("--online_steps", default=1e5, type=int)    # For online training and offline-to-online fine-tuning
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)                 # Discount factor
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--expectile", default=0.7, type=float)           # Hyperparameter for expectile regression
    parser.add_argument("--temperature", default=3.0, type=float)             # temperature for action/state weighting
    parser.add_argument("--num_samples", default=20, type=int)     # number of sampled noises
    parser.add_argument("--v", action="store_true")                 # whether to maximize value at next state
    # Work dir
    parser.add_argument('--work_dir', default='tmp', type=str)      # Work dir to log
    parser.add_argument('--save_video', default=False, action='store_true') # Whether to save video
    args = parser.parse_args()
    args.cooldir = generate_slug(2)

    # Build work dir
    base_dir = 'runs'
    utils.make_dir(base_dir)
    base_dir = os.path.join(base_dir, args.work_dir)
    utils.make_dir(base_dir)
    args.work_dir = os.path.join(base_dir, args.env)
    utils.make_dir(args.work_dir)

    # make directory
    ts = time.gmtime()
    ts = time.strftime("%m-%d-%H:%M", ts)
    exp_name = str(args.env) + '-' + ts + '-bs' + str(args.batch_size) + '-s' + str(args.seed)
    if args.policy == 'SAW':
        exp_name += '-expectile' + str(args.expectile) + '--temp' + str(args.temperature)
    else:
        raise NotImplementedError
    exp_name += '-' + args.cooldir
    args.work_dir = args.work_dir + '/' + exp_name
    utils.make_dir(args.work_dir)

    args.model_dir = os.path.join(args.work_dir, 'model')
    utils.make_dir(args.model_dir)
    args.video_dir = os.path.join(args.work_dir, 'video')
    utils.make_dir(args.video_dir)

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    utils.snapshot_src('.', os.path.join(args.work_dir, 'src'), '.gitignore')

    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    if hasattr(env.action_space, 'n'):
        action_shape = env.action_space.n
    else:
        action_shape = env.action_space.shape[0]
    
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "expectile": args.expectile,
        "temperature": args.temperature,
        "num_samples": args.num_samples,
        "v": args.v,
    }

    # Initialize policy
    if args.policy == 'SAW':
        policy = SAW.SAW(**kwargs)
    else:
        raise NotImplementedError

    if args.load_model != "":
        policy.load(f"./{args.load_model}")

    # offline training
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    dataset = d4rl.qlearning_dataset(env)
    states = dataset['observations']
    replay_buffer.convert_D4RL(dataset)
    if 'antmaze' in args.env:
        # Center reward for Ant-Maze
        # See https://github.com/aviralkumar2907/CQL/blob/master/d4rl/examples/cql_antmaze_new.py#L22
        replay_buffer.reward = replay_buffer.reward - 1.0
    min_state, max_state = replay_buffer.normalize_states()

    logger = Logger(args.work_dir, use_tb=True)
    video = VideoRecorder(dir_name=args.video_dir)

    #policy.warmup(replay_buffer, None, args.batch_size)

    for t in trange(int(args.max_timesteps)):
        policy.train(replay_buffer, None, args.batch_size, logger=logger, offline=True)
        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            eval_episodes = 100 if 'antmaze' in args.env else 10
            d4rl_score = eval_policy(args, t+1, video, logger, policy, args.env,
                                    args.seed, min_state, max_state, eval_episodes=eval_episodes, offline=True)
            if args.save_model:
                policy.save(args.model_dir)
