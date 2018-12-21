import argparse
import gym
import banana

def parse_arguments():
    parser = argparse.ArgumentParser(description="Rainbow DQN based navigation project")
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

    # Create environment
    env = gym.make("Banana-v1")
    env.render('realtime')

    state = env.reset()
    score = 0
    done = False
    while not done:
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        score += reward
        state = next_state
    print(env)
    # brain_name = env.brain_names[0]
    # brain = env.brains[brain_name]
    # env_info = env.reset(train_mode=True)[brain_name]
    # action_size = brain.vector_action_space_size
    #
    # # examine the state space
    # state = env_info.vector_observations[0]
    # print('States look like:', state)
    # state_size = len(state)
    # print('States have length:', state_size)
    #
    # # Create rainbow DQN learning agent
    # env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
    #
    # state = env_info.vector_observations[0]  # get the current state
    # score = 0  # initialize the score
    # while True:
    #     action = np.random.randint(action_size)  # select an action
    #     env_info = env.step(action)[brain_name]  # send the action to the environment
    #     next_state = env_info.vector_observations[0]  # get the next state
    #     reward = env_info.rewards[0]  # get the reward
    #     done = env_info.local_done[0]  # see if episode has finished
    #     score += reward  # update the score
    #     state = next_state  # roll over the state to next time step
    #     if done:  # exit loop if episode finished
    #         break


if __name__ == '__main__':
    main()
