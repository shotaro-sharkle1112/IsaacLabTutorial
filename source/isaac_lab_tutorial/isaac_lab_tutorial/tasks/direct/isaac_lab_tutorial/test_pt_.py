# predict_from_skrl_checkpoint.py
from test_pt import predict_actions

obs = [ 1.0000, -0.0060,  1.0015]
action = predict_actions(obs, deterministic=True)
print(action)