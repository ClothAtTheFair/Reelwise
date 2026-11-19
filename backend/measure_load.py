import os
import psutil
import joblib
import time

process = psutil.Process(os.getpid())

def memory_gb():
    return process.memory_info().rss / (1024 ** 3)

print(f"Before loading: {memory_gb():.2f} GB")

start = time.perf_counter()

model = joblib.load("models/hybrid.joblib")

print(f"After loading: {memory_gb():.2f} GB")
print(f"Load time: {time.perf_counter() - start:.2f}s")
