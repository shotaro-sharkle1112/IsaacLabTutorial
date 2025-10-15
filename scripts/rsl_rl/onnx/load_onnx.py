import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("policy.onnx", providers=["CPUExecutionProvider"])

# 入出力名の取得
in_name  = sess.get_inputs()[0].name
out_names = [o.name for o in sess.get_outputs()]

print("in_name",in_name)
print("out_name",out_names)