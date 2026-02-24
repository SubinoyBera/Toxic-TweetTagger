from tests.load_test.locust import HttpUser, task, between
import random

# Load comments once at startup
with open("comments.txt", "r", encoding="utf-8") as f:
    COMMENTS = [line.strip() for line in f if line.strip()]


class InferenceUser(HttpUser):
    wait_time = between(0, 0)

    @task
    def predict(self):
        text = random.choice(COMMENTS)

        payload = {
            "comment": text
        }

        self.client.post(
            "/get_prediction",
            json=payload
        )