from skrl.agents.torch.ppo import PPO

# Instantiate the agent
agent = PPO(models=models,  # models dict
            memory=memory,  # memory instance, or None if not required
            cfg=agent_cfg,  # configuration dict (preprocessors, learning rate schedulers, etc.)
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device)

# Load the checkpoint
agent.load("./runs/22-09-29_22-48-49-816281_DDPG/checkpoints/agent_1200.pt")