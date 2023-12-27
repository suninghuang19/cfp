import time
import queue
import random
import threading
import numpy as np

class ReplayMemory:
    def __init__(self, capacity, seed, batch_size=256):
        random.seed(seed)
        self.length = 0
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.cache = queue.Queue(maxsize=1)
        self.cache_batch_size = batch_size
        self.worker = None

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        self.length = min(self.length + 1, self.capacity)

    def sample(self, batch_size=256):
        if self.worker is None:
            self.cache_batch_size = batch_size
            # parallel sampling to speed up
            self.worker = threading.Thread(target=self.sample_inf, daemon=True)
            self.worker.start()

        if self.cache_batch_size != batch_size:
            self.cache_batch_size = batch_size
            _ = self.cache.get()

        samples = self.cache.get()
        self.cache_batch_size = batch_size
        return samples

    def sample_inf(self):
        while True:
            if self.cache.empty() and len(self.buffer) >= self.cache_batch_size:
                batch = random.sample(self.buffer, self.cache_batch_size)
                samples = map(np.stack, zip(*batch))
                if self.cache.full():
                    self.cache.get()
                self.cache.put(samples)
            else:
                time.sleep(0.005)

    def __len__(self):
        return self.length
