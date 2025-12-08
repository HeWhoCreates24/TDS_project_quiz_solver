#!/usr/bin/env python3
"""
Quick answer injection script for manual override during quiz solving.
Edit the ANSWER variable below and run this script to add your answer to the queue.
The solver will try it on whatever quiz it's currently solving.
"""

import requests
import json

# ========== CONFIGURATION ==========
SECRET = "my_secret_key"
OVERRIDE_ENDPOINT = "http://localhost:8080/override"

# ========== EDIT YOUR ANSWER HERE ==========
ANSWER = 42  # Change this to your answer (can be string, number, list, dict, etc.)
# Examples:
# ANSWER = "literal_test_value"
# ANSWER = 42
# ANSWER = [1, 2, 3]
# ANSWER = {"key": "value"}
# ========================================

def inject_answer():
    """Add manual answer to the queue - will be tried on current quiz."""
    payload = {
        "secret": SECRET,
        "answer": ANSWER
    }
    
    print(f"\n{'='*60}")
    print("INJECTING MANUAL ANSWER TO QUEUE")
    print(f"{'='*60}")
    print(f"Answer: {ANSWER}")
    print(f"Answer Type: {type(ANSWER).__name__}")
    print(f"{'='*60}\n")
    
    try:
        response = requests.post(
            OVERRIDE_ENDPOINT,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✓ SUCCESS!")
            print(f"Message: {result.get('message', 'Answer queued')}")
            print(f"Queue Position: {result.get('queue_position', 'unknown')}")
            print(f"\n{result}")
        else:
            print(f"✗ FAILED!")
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")
    
    except Exception as e:
        print(f"✗ ERROR!")
        print(f"Exception: {e}")

if __name__ == "__main__":
    inject_answer()
