README.md — DQN Atari Pong Project

Overview

This project implements Deep Q-Learning (DQN) and two major extensions — Double DQN and Dueling DQN — on the Atari game Pong (ALE/Pong-v5).

The goals of the project:

Build a full baseline DQN agent

Add at least one extension (I included both Double + Dueling)

Train the agent with replay buffers, frame-stacking, and target networks

Track hyperparameters, experiments, curves

Record gameplay videos (early/random vs. trained)

Produce a short reflection on challenges and improvements

This repository contains:

Training notebook(s)

Model code (DQN, Double DQN, Dueling DQN)

Environment wrappers

Learning curves

Videos (random & trained)

PDF report + slides

Starter code adapted from the assignment notebook:
c166f25_02b_dqn_pong.ipynb



1:Environment: ALE/Pong-v5

Observation space:

After preprocessing, each observation is a stack of 4 grayscale 84×84 frames.

Shape: (4, 84, 84)

Type: uint8

Purpose: captures motion across time for the ball + paddle.

Action space:

Discrete(6) Atari actions (NOOP, FIRE, RIGHT, LEFT, UP, DOWN depending on Pong config).

For Pong, only 2–3 actions matter, but agent still must choose from all 6.

Reward structure & quirks:

+1 when the agent scores a point.

–1 when the agent loses a point.

All rewards are sparse and extremely delayed compared to frame rate (~60 FPS).

Long stretches of zero reward → makes exploration difficult early in training.

Episode dynamics:

Each Pong game is ~20–21 points.

A poorly performing policy gets almost always –21 early on.

Episodes can be long, so training is dominated by exploration + credit assignment.


2:Hyperparameters

Common training setup

Component	Value / Choice

Environment = 	ALE/Pong-v5

Observation = 	4× stacked 84×84 grayscale frames (uint8)

Action space =	Discrete(6)

Optimizer =	Adam

Learning rate =	1e-4

Discount factor γ =	0.99

Replay buffer size =	10,000 transitions

Replay warm-up =	10,000 steps before starting updates

Batch size =	32

Target network sync =	Every 1,000 environment steps

Frame stack =	4 frames

Max episodes (runs shown) =	~80–130 episodes (early-stage curves)



Exploration schedule (ε-greedy)

Parameter =	Value
ε start =	1.0
ε final =	0.02
Decay horizon =	200,000 steps (linear)
Policy =	ε-greedy over Q(s,·)


Baseline vs. Dueling+Double runs


Setting				Baseline DQN			Dueling + Double DQN
Q-network head			Conv → FC →			A

Target update rule		Standard DQN:max_a Q̂(s',a)	Double DQN:argmax from online net,value from target

Loss				MSE between Q(s,a) and 1-step TD target		Same, but using Double-DQN target

All other hyperparams		Same as common setup above	Same as common setup above


3:Experimentation Log (Targeted Hyperparameter Tuning) 5-8 targeted changes


1. Lowered the learning rate from 2.5e-4 → 1e-4

Why: Atari DQN papers note instability with high LR early in training.

Effect: Training became less noisy; moving average curve smoothed out slightly.


2. Increased target network sync frequency from 5,000 → 1,000 steps

Why: Pong is sensitive to stale targets during early learning.

Effect: Reduced large spikes in TD loss; reward curve became more consistent.

3. Adjusted ε-decay horizon from 1e5 → 2e5 steps

Why: Pong has sparse/delayed rewards; needs longer exploration.

Effect: Slower drop in ε improved initial exploration and slightly raised early reward spikes.

4. Increased replay warmup from 1,000 → 10,000 steps

Why: Learning too early from tiny replay buffer produces garbage gradients.

Effect: Stabilized the first 5–10 episodes and removed “random collapse” behavior.

5. Switched from vanilla DQN → Double DQN (2nd run)

Why: Reduces overestimation bias in action values.

Effect: Q-values became more stable; targets less noisy. Small improvement in consistency.

4:Learning Curves

Plots included inside the notebook:

Episode reward vs. episode

100-episode moving average

Baseline DQN curve

Dueling + Double curve

Expect early baseline rewards around –21 to –18.
Variants reduce noise but need longer training to show major improvements.

5:Videos

Two videos recorded as required:

1. Random Policy (Early / Untrained)

Shows baseline random paddle movement, consistently loses.

pong_random_baseline.mp4

2. “Trained” Policy (after 300–400k frames)

Agent is still early-stage but slightly more consistent and reacts more to the ball.

pong_trained_baseline.mp4


6:Requirements

gymnasium
ale-py
stable-baselines3
torch
numpy
matplotlib
autorom

7:How to Run

Clone the repo:

git clone <your-link>
cd dqn-pong


Install deps:

pip install -r requirements.txt


Run the notebook:

jupyter notebook

8:Reflection - I chose Pong because it is the most widely studied benchmark for value-based deep reinforcement learning, and it gives a clean way to compare the behavior of baseline DQN against improved variants such as Double DQN and Dueling DQN. Pong also highlights the challenges that arise from sparse and delayed rewards. Since the agent receives almost no useful signal for long periods of time, the learning process is extremely slow without massive compute budgets or advanced replay strategies. For this reason, my baseline model performed similarly to a random policy for a large portion of training, which is consistent with the original DQN literature.

The main challenge of this project was stabilizing learning in a setting where Q-values are noisy and easily overestimated. Adding Double DQN reduced this overestimation by separating action selection from action evaluation, which led to slightly more stable learning curves. The Dueling DQN architecture also improved value estimation by separating state-value from action advantage, which helped the model produce more consistent Q-values even when there was little movement in the environment. While neither model “mastered” Pong within the limited training budget available on Colab, both showed improvements over the baseline in terms of variance reduction and smoother learning signals.

If I continued this work, the next steps would be to incorporate prioritized experience replay (PER), n-step returns, sticky actions, and slower epsilon decay to help the agent better explore and propagate reward signals. I would also increase the total frame budget to millions of frames, which is what successful Atari agents typically require. With these refinements, I would expect the agent to progress beyond the random-play phase and begin developing stable Pong strategies such as tracking the ball and returning volleys. Even without full convergence, the project clearly showed the benefits of Double and Dueling DQN in stabilizing value-based RL.
