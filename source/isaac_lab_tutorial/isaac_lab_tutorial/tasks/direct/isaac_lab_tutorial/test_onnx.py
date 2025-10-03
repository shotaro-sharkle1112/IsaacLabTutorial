import onnxruntime as ort
import numpy as np

POLICY_PATH = r"C:\Users\xr\Issac\IssacLab\Projects\IsaacLabTutorial\source\isaac_lab_tutorial\isaac_lab_tutorial\tasks\direct\isaac_lab_tutorial\policy.onnx"
OBS_SHAPE = (3,)     # 例: CartPole-v1 なら観測 4 次元。ご自身の環境に合わせて！
ACTION_DIM = 2

session = ort.InferenceSession(POLICY_PATH, providers=["CPUExecutionProvider"])

x = np.random.randn(3, *OBS_SHAPE).astype(np.float32)
pm, pls = session.run(None, {"obs": x})

print("ONNX policy shapes:", pm.shape, pls.shape)
