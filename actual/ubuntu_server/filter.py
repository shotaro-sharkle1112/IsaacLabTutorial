import json
import argparse

LOW, HIGH = -0.15, 0.18

def in_range(x: float, low: float = LOW, high: float = HIGH) -> bool:
    return low <= x <= high

def keep_record(rec: dict) -> bool:
    obs0 = rec.get("obs", [None])[0]
    next_obs0 = rec.get("next_obs", [None])[0]
    print("--------------------")
    print(obs0)
    print(next_obs0)
    print(in_range(obs0) and in_range(next_obs0))
    if obs0 is None or next_obs0 is None:
        return False
    return in_range(obs0) and in_range(next_obs0)

def filter_jsonl(input_path: str, output_path: str) -> None:
    kept = 0
    total = 0

    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                # 壊れた行があっても止めたくない場合はスキップ
                # 必要なら print で警告を出してもOK
                continue

            if keep_record(rec):
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1

    print(f"done. total={total}, kept={kept}, removed={total - kept}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter jsonl by obs[0] or next_obs[0] in [-0.09, 0.09].")
    parser.add_argument("input", help="input jsonl path")
    parser.add_argument("output", help="output jsonl path")
    args = parser.parse_args()

    filter_jsonl(args.input, args.output)
