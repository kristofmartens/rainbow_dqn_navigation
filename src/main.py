import argparse
import gym
import banana
import random
import torch
import os
from collections import deque

from rainbow_dqn_agent import DQNAgent, ReplayMemory


def parse_arguments():
    parser = argparse.ArgumentParser(description="Rainbow DQN based navigation project")
    parser.add_argument('--seed', type=int, default=0, help="Random seed")
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
    parser.add_argument('--realtime', action='store_true', help="Run code in real-time")
    parser.add_argument('--evaluation', action='store_true', help="Don't train by default")
    parser.add_argument('--evaluation-episodes', type=int, default=100,
                        help="Number of episodes to average evaluation over")
    parser.add_argument('--replay-frequency', type=int, default=1, help="Frequency of sampling from memory")
    parser.add_argument('--reward-clip', type=int, default=0, help="Value for clipping the reward, 0 to disable")
    parser.add_argument('--target-update', type=int, default=800, help="Update target network after this many steps")
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
    if args.realtime is True:
        env.render('realtime')
    else:
        env.render('training')

    evaluation_buffer = deque(maxlen=args.evaluation_episodes)

    # Create agent
    agent = DQNAgent(env,
                     gamma=args.discount,
                     batch_size=args.batch_size,
                     device=args.device,
                     learning_rate=args.adam_learning_rate,
                     epsilon=args.adam_epsilon)
    memory = ReplayMemory(capacity=args.memory_capacity)

    # Load model from memory
    if args.evaluation is True:
        agent.load(os.path.join(os.getcwd(), "result"))

    step = 0
    done = True
    episode = 0
    score = 0

    agent.train()  # Set agent in training mode
    while step < args.max_steps:
        if done:
            state = torch.Tensor(env.reset()).view(-1, env.observation_space.shape[0])
            score = 0
            episode += 1

        # Choose an action
        agent.eval()  # Set agent to evalutation mode
        if args.evaluation is True:
            action = agent.act(state)
        else:
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
        if args.evaluation is False and step > args.learning_start:
            agent.train()  # Set agent to training mode
            # Learn at replay frequency steps
            if step % args.replay_frequency == 0:
                agent.learn(memory)

            # Update target network for Double Q learning
            if step % args.target_update == 0:
                agent.update_target_net()

            if step % 10000 == 0:
                agent.save(os.path.join(os.getcwd(), "result"))

        # Update state
        state = next_state

        if done:
            evaluation_buffer.append(score)
            print("Episode: {}\tStep: {}\tScore: {}\tAvg Score: {}".format(episode, step, score,
                                                                           sum(evaluation_buffer) / len(evaluation_buffer)))


if __name__ == '__main__':
    main()
