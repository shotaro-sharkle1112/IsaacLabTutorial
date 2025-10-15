import numpy as np
import onnxruntime as ort

# 例の obs 値
obs = np.array([0.0002243409544462338, -0.0006787014426663518, 15.614333152770996, -0.9833188056945801], dtype=np.float32)

sess = ort.InferenceSession("policy.onnx", providers=["CPUExecutionProvider"])

in_name = sess.get_inputs()[0].name     # 入力名（例: 'obs' や 'input'）
out_names = [o.name for o in sess.get_outputs()]  # 出力名（1つだけのはず）

# 多くのモデルは [batch, 4] を想定するのでバッチ次元を付ける
obs_batched = obs.reshape(1, 4)

outputs = sess.run(out_names, {in_name: obs_batched})

# 出力をスカラー/1次元に整形
action = float(np.asarray(outputs[0]).squeeze())
print("action:", action)