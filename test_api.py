#!/usr/bin/env python3
"""
Test script for the Iris Model API
"""
import requests
import json
import time
import subprocess
import sys
from multiprocessing import Process


def start_server():
    """Start the FastAPI server in background"""
    subprocess.run([sys.executable, "app.py"])


def test_api():
    """Test the API endpoints"""
    base_url = "http://localhost:8000"

    # Wait for server to start
    print("⏳ Waiting for server to start...")
    time.sleep(3)

    try:
        # Test root endpoint
        print("🧪 Testing root endpoint...")
        response = requests.get(f"{base_url}/")
        print(f"✅ Root endpoint: {response.json()}")

        # Test prediction endpoint with different iris samples
        test_cases = [
            {
                "features": [5.1, 3.5, 1.4, 0.2],
                "expected_class": "Setosa",
                "description": "Typical Setosa sample",
            },
            {
                "features": [6.2, 2.9, 4.3, 1.3],
                "expected_class": "Versicolor",
                "description": "Typical Versicolor sample",
            },
            {
                "features": [7.3, 2.9, 6.3, 1.8],
                "expected_class": "Virginica",
                "description": "Typical Virginica sample",
            },
        ]

        class_names = ["Setosa", "Versicolor", "Virginica"]

        print("\n🧪 Testing prediction endpoint...")
        for i, test_case in enumerate(test_cases, 1):
            response = requests.post(
                f"{base_url}/predict",
                json={"features": test_case["features"]},
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                result = response.json()
                predicted_class = class_names[result["prediction"]]
                print(f"✅ Test {i}: {test_case['description']}")
                print(f"   Features: {test_case['features']}")
                print(f"   Predicted: {predicted_class} (class {result['prediction']})")
                print(f"   Expected: {test_case['expected_class']}")
                print()
            else:
                print(f"❌ Test {i} failed: {response.status_code} - {response.text}")

        # Test error handling
        print("🧪 Testing error handling...")
        response = requests.post(
            f"{base_url}/predict",
            json={"features": [1, 2]},  # Wrong number of features
            headers={"Content-Type": "application/json"},
        )

        if response.status_code != 200:
            print("✅ Error handling works correctly")
        else:
            print("⚠️  API should reject invalid input")

        print("\n🎉 All tests completed!")

    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the server is running.")
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")


if __name__ == "__main__":
    print("🚀 Starting API Test Suite")
    print("=" * 40)

    # Check if server is already running
    try:
        response = requests.get("http://localhost:8000/", timeout=1)
        print("✅ Server is already running")
        test_api()
    except requests.exceptions.ConnectionError:
        print("🔄 Starting server...")
        # Start server in background process
        server_process = Process(target=start_server)
        server_process.start()

        try:
            test_api()
        finally:
            print("🛑 Stopping server...")
            server_process.terminate()
            server_process.join()

    print("\n✨ Test suite finished!")
