import argparse
import gym
import banana
import random
import torch

from rainbow_dqn_agent import DQNAgent, ReplayMemory


def parse_arguments():
    parser = argparse.ArgumentParser(description="Rainbow DQN based navigation project")
    parser.add_argument('--seed', type=int, default=0, help="Random seed")
    parser.add_argument('--model', type=str, help='Pretrained model (state dict)')
    parser.add_argument('--discount', type=float, default=0.99, help='Discount rate')
    parser.add_argument('--batch-size', type=int, default=16, help='Training batch size')
    parser.add_argument('--adam-learning-rate', type=float, default=1e-5, help="ADAM Training learning rate")
    parser.add_argument('--adam-epsilon', type=float, default=1e-8, help="Adam Epsilon parameter")
    parser.add_argument('--multi-step', type=int, default=1, help="Number of steps for multi-step return")
    parser.add_argument('--value-dist-size', type=int, default=51, help="Discretised size of value distribution")
    parser.add_argument('--vmax', type=int, default=2, help="Maximum value distribution support")
    parser.add_argument('--vmin', type=int, default=-2, help="Minimum value distribution support")
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu', help="Device where to train")
    parser.add_argument('--memory-capacity', type=int, default=int(3e5), help="Size of the replay buffer")
    parser.add_argument('--priority-weight', type=float, default=0.4,
                        help="Initial prioritised experience replay importance sampling weight")
    parser.add_argument('--priority-exponent', type=float, default=0.5, help="Prioritised experience replay exponent")
    parser.add_argument('--max-steps', type=int, default=1e7, help="Maximum number of episodes")
    parser.add_argument('--epsilon-greedy-steps', type=int, default=5e4, help="Initial steps to epsilon greedy explore")
    parser.add_argument('--learning-start', type=int, default=500, help="Number of iterations before start training")
    parser.add_argument('--evaluation-episodes', type=int, default=10,
                        help="Number of episodes to average evaluation over")
    parser.add_argument('--evaluation-size', type=int, default=500, help="Number of transitions to validate Q-value")
    parser.add_argument('--replay-frequency', type=int, default=1, help="Frequency of sampling from memory")
    parser.add_argument('--reward-clip', type=int, default=0, help="Value for clipping the reward, 0 to disable")
    parser.add_argument('--target-update', type=int, default=16 * 50,
                        help="Update target network after this many steps")

    # parser.add_argument('--seed', type=int, default=123, help='Random seed')
    # parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    # parser.add_argument('--game', type=str, default='space_invaders', help='ATARI game')
    # parser.add_argument('--T-max', type=int, default=int(50e6), metavar='STEPS',
    #                     help='Number of training steps (4x number of frames)')
    # parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH',
    #                     help='Max episode length (0 to disable)')
    # parser.add_argument('--history-length', type=int, default=4, metavar='T',
    #                     help='Number of consecutive states processed')
    # parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')
    # parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ',
    #                     help='Initial standard deviation of noisy linear layers')
    # parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
    # parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
    # parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
    # parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
    # parser.add_argument('--memory-capacity', type=int, default=int(1e6), metavar='CAPACITY',
    #                     help='Experience replay memory capacity')
    # parser.add_argument('--replay-frequency', type=int, default=4, metavar='k',
    #                     help='Frequency of sampling from memory')
    # parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω',
    #                     help='Prioritised experience replay exponent (originally denoted α)')
    # parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β',
    #                     help='Initial prioritised experience replay importance sampling weight')
    # parser.add_argument('--multi-step', type=int, default=3, metavar='n', help='Number of steps for multi-step return')
    # parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
    # parser.add_argument('--target-update', type=int, default=int(32e3), metavar='τ',
    #                     help='Number of steps after which to update target network')
    # parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE', help='Reward clipping (0 to disable)')
    # parser.add_argument('--lr', type=float, default=0.0000625, metavar='η', help='Learning rate')
    # parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
    # parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
    # parser.add_argument('--learn-start', type=int, default=int(80e3), metavar='STEPS',
    #                     help='Number of steps before starting training')
    # parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
    # parser.add_argument('--evaluation-interval', type=int, default=100000, metavar='STEPS',
    #                     help='Number of training steps between evaluations')
    # parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N',
    #                     help='Number of evaluation episodes to average over')
    # parser.add_argument('--evaluation-size', type=int, default=500, metavar='N',
    #                     help='Number of transitions to use for validating Q')
    # parser.add_argument('--log-interval', type=int, default=25000, metavar='STEPS',
    #                     help='Number of training steps between logging status')
    # parser.add_argument('--render', action='store_true', help='Display screen (testing only)')
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Setup torch
    random.seed(args.seed)
    torch.manual_seed(random.randint(1, 10000))

    if torch.cuda.is_available() and args.device is 'cuda':
        torch.cuda.manual_seed(random.randint(1, 10000))
        torch.backends.cudnn.enabled = False
    else:
        args.device = 'cpu'

    # Create environment
    env = gym.make("Banana-v1")
    env.render('training')

    # Create agent
    agent = DQNAgent(env,
                     gamma=args.discount,
                     batch_size=args.batch_size,
                     device=args.device,
                     learning_rate=args.adam_learning_rate)
    memory = ReplayMemory(capacity=args.memory_capacity)

    agent.train()  # Set agent in training mode
    step = 0
    done = True
    episode = 0
    score = 0

    while step < args.max_steps:
        if done:
            state = torch.Tensor(env.reset()).view(-1, env.observation_space.shape[0])
            score = 0
            episode += 1

        # # Draw new set of noisy weights
        # if step % args.target_update == args.target_update/2:
        #     agent.reset_noise()

        # Choose an action
        agent.eval()  # Set agent to evalutation mode
        epsilon = min(step / args.epsilon_greedy_steps, 0.95)
        action = agent.act_e_greedy(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        next_state = torch.Tensor(next_state).view(-1, env.observation_space.shape[0])

        # Clip rewards
        if args.reward_clip > 0:
            reward = max(min(reward, args.reward_clip), -args.reward_clip)

        # Score
        score += reward

        # Add to replay memory
        memory.append(state, action, reward, next_state, done)

        # Increase the step
        step += 1

        # Wait to start training
        if step > args.learning_start:
            agent.train()  # Set agent to training mode
            # Learn at replay frequency steps
            if step % args.replay_frequency == 0:
                agent.learn(memory)

            # Update target network for Double Q learning
            if step % args.target_update == 0:
                agent.update_target_net()

            if step % 10000 == 0:
                agent.save("/Users/kristofmartens/project/reinforcement_learning/rainbow_dqn_navigation/result")

        # Update state
        state = next_state

        if done:
            print("Episode: {}\tStep: {}\tScore: {}".format(episode, step, score))


if __name__ == '__main__':
    main()
