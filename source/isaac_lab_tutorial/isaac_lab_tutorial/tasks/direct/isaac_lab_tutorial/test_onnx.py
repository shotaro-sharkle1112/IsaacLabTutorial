import onnxruntime as ort
import numpy as np

POLICY_PATH = r"C:\Users\xr\Issac\IssacLab\Projects\IsaacLabTutorial\source\isaac_lab_tutorial\isaac_lab_tutorial\tasks\direct\isaac_lab_tutorial\policy.onnx"
OBS_SHAPE = (3,)     # 例: CartPole-v1 なら観測 4 次元。ご自身の環境に合わせて！
ACTION_DIM = 2

session = ort.InferenceSession(POLICY_PATH, providers=["CPUExecutionProvider"])

x = np.array([[ 1.0000, -0.0060,  1.0015]]).astype(np.float32)
pm, pls = session.run(None, {"obs": x})

print("ONNX policy shapes:", pm.shape, pls.shape)

print("pm",pm[0])


groud_truth ="""
[DEBUG]: left 2.2136287689208984, right 2.2373287677764893
[DEBUG]: obs tensor([ 1.0000, -0.0060,  1.0015], device='cuda:0')
[DEBUG]: left 2.213836669921875, right 2.237321138381958
[DEBUG]: obs tensor([ 1.0000, -0.0060,  1.0014], device='cuda:0')
[DEBUG]: left 2.2140116691589355, right 2.2373154163360596
[DEBUG]: obs tensor([ 1.0000, -0.0061,  1.0015], device='cuda:0')
[DEBUG]: left 2.2141082286834717, right 2.2372372150421143
[DEBUG]: obs tensor([ 1.0000, -0.0061,  0.9994], device='cuda:0')
[DEBUG]: left 2.2163138389587402, right 2.2392475605010986
[DEBUG]: obs tensor([ 1.0000, -0.0061,  1.0010], device='cuda:0')
[DEBUG]: left 2.214837074279785, right 2.237671375274658
[DEBUG]: obs tensor([ 1.0000, -0.0061,  0.9999], device='cuda:0')
[DEBUG]: left 2.216000556945801, right 2.2387051582336426
[DEBUG]: obs tensor([ 1.0000, -0.0061,  0.9996], device='cuda:0')
[DEBUG]: left 2.2163825035095215, right 2.2389769554138184
[DEBUG]: obs tensor([ 1.0000, -0.0061,  1.0006], device='cuda:0')
[DEBUG]: left 2.2154626846313477, right 2.2379655838012695
[DEBUG]: obs tensor([ 1.0000, -0.0061,  1.0005], device='cuda:0')
[DEBUG]: left 2.215726613998413, right 2.238121747970581
[DEBUG]: obs tensor([ 1.0000, -0.0061,  1.0002], device='cuda:0')
[DEBUG]: left 2.216045618057251, right 2.238347291946411
[DEBUG]: obs tensor([ 1.0000, -0.0061,  1.0000], device='cuda:0')
[DEBUG]: left 2.216344118118286, right 2.2385592460632324
[DEBUG]: obs tensor([ 1.0000, -0.0061,  1.0001], device='cuda:0')
[DEBUG]: left 2.2162766456604004, right 2.2384347915649414
[DEBUG]: obs tensor([ 1.0000, -0.0062,  1.0023], device='cuda:0')
[DEBUG]: left 2.2142446041107178, right 2.236278533935547
[DEBUG]: obs tensor([ 1.0000, -0.0062,  1.0017], device='cuda:0')
[DEBUG]: left 2.2149696350097656, right 2.236865282058716
[DEBUG]: obs tensor([ 1.0000, -0.0062,  1.0021], device='cuda:0')
[DEBUG]: left 2.214639663696289, right 2.236407518386841
[DEBUG]: obs tensor([ 1.0000, -0.0062,  1.0019], device='cuda:0')
[DEBUG]: left 2.214932918548584, right 2.236567258834839
[DEBUG]: obs tensor([ 1.0000, -0.0062,  1.0018], device='cuda:0')
[DEBUG]: left 2.2151927947998047, right 2.2367300987243652
[DEBUG]: obs tensor([ 1.0000, -0.0062,  1.0018], device='cuda:0')
[DEBUG]: left 2.2152633666992188, right 2.2366857528686523
[DEBUG]: obs tensor([ 1.0000, -0.0062,  1.0016], device='cuda:0')
[DEBUG]: left 2.215487480163574, right 2.2368109226226807
[DEBUG]: obs tensor([ 1.0000, -0.0062,  1.0016], device='cuda:0')
[DEBUG]: left 2.2155919075012207, right 2.2368242740631104
[DEBUG]: obs tensor([ 1.0000, -0.0062,  0.9992], device='cuda:0')
[DEBUG]: left 2.218069076538086, right 2.2391695976257324
[DEBUG]: obs tensor([ 1.0000, -0.0062,  1.0018], device='cuda:0')
[DEBUG]: left 2.2156026363372803, right 2.2366323471069336
[DEBUG]: obs tensor([ 1.0000, -0.0062,  1.0001], device='cuda:0')
[DEBUG]: left 2.21734619140625, right 2.238262176513672
[DEBUG]: obs tensor([ 1.0000, -0.0063,  1.0002], device='cuda:0')
[DEBUG]: left 2.2173070907592773, right 2.2381432056427
[DEBUG]: obs tensor([ 1.0000, -0.0063,  1.0012], device='cuda:0')
[DEBUG]: left 2.2164359092712402, right 2.2372097969055176
[DEBUG]: obs tensor([ 1.0000, -0.0063,  1.0010], device='cuda:0')
[DEBUG]: left 2.216604232788086, right 2.2373156547546387
[DEBUG]: obs tensor([ 1.0000, -0.0063,  1.0009], device='cuda:0')
[DEBUG]: left 2.2167704105377197, right 2.2374267578125
[DEBUG]: obs tensor([ 1.0000, -0.0063,  1.0006], device='cuda:0')
[DEBUG]: left 2.2171199321746826, right 2.23773193359375
[DEBUG]: obs tensor([ 1.0000, -0.0063,  1.0006], device='cuda:0')
[DEBUG]: left 2.217174530029297, right 2.237755537033081
[DEBUG]: obs tensor([ 1.0000, -0.0063,  1.0023], device='cuda:0')
[DEBUG]: left 2.215524673461914, right 2.2359907627105713
[DEBUG]: obs tensor([ 1.0000, -0.0063,  1.0021], device='cuda:0')
[DEBUG]: left 2.215867042541504, right 2.236211061477661
"""