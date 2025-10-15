import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("policy.onnx", providers=["CPUExecutionProvider"])

# 入出力名の取得
in_name  = sess.get_inputs()[0].name
out_names = [o.name for o in sess.get_outputs()]

print("Inputs:")
for i, inp in enumerate(sess.get_inputs()):
    print(i, inp.name, inp.shape, inp.type)
print("Outputs:")
for i, out in enumerate(sess.get_outputs()):
    print(i, out.name, out.shape, out.type)