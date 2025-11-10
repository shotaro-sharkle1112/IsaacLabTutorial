#!/usr/bin/env python3
import requests
import json
SERVER_URL = "http://192.168.3.12:5000"

def test_health():
    print("===health check===")
    response = requests.get(f"{SERVER_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}\n")


def test_inference(observation):
    print(f"=== inference test ===")
    print(f"post data: {observation}")

    response = requests.post(
        f"{SERVER_URL}/infer",
        json={"observation": observation},
        headers={"Content-Type": "application/json"}
    )

    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}\n")
    return response.json()


if __name__ == "__main__":
    test_health()
    sample_observations = [
        [0.0734, -0.0573, 0.0016, 0.2029],
        [ 0.0692, -0.2256,  0.0054,  0.2294],
        [ 0.0679, -0.0344,  0.0071,  0.0662],
    ]

    for obs in sample_observations:
        result = test_inference(obs)
        if "action" in result:
            print(f"inference result: {result['action']}\n")
