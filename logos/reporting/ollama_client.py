import requests


class OllamaClient:

    def __init__(
        self,
        model="qwen3:1.7b",
        host="http://localhost:11434"
    ):
        self.model = model
        self.host = host

    def generate(self, prompt):

        print("HOST:", self.host)
        print("MODEL:", self.model)

        response = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=600,
        )

        print("STATUS:", response.status_code)
        print("RESPONSE:", response.text[:1000])

        response.raise_for_status()

        return response.json()["response"]