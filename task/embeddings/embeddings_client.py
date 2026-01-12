import json

import requests

from task._constants import API_KEY

DIAL_EMBEDDINGS = 'https://ai-proxy.lab.epam.com/openai/deployments/{model}/embeddings'


#TODO:
# ---
# https://dialx.ai/dial_api#operation/sendEmbeddingsRequest
# ---
# Implement DialEmbeddingsClient:
# - constructor should apply deployment name and api key
# - create method `get_embeddings` that will generate embeddings for input list (don't forget about dimensions)
#   with Embedding model and return back a dict with indexed embeddings (key is index from input list and value vector list)

class DialEmbeddingsClient:
    ...

    def __init__(self, deployment_name: str, api_key: str = API_KEY):
        self.deployment_name = deployment_name
        self.api_key = api_key
        self.url = DIAL_EMBEDDINGS.format(model=self.deployment_name)
        self.headers = {
            'Content-Type': 'application/json',
            'api-key': self.api_key
        }

    def get_embeddings(self, input_list: list, dimensions: int) -> dict[int, list[float]]:
        payload = {
            "input": input_list,
            "dimensions": dimensions
        }
        response = requests.post(self.url, headers=self.headers, json=payload)

        if response.status_code != 200:
            raise Exception(f"Error fetching embeddings: {response.status_code}, {response.text}")

        response_json = response.json()

        return {item['index']: item['embedding'] for item in response_json['data']}
# Hint:
#  Response JSON:
#  {
#     "data": [
#         {
#             "embedding": [
#                 0.19686688482761383,
#                 ...
#             ],
#             "index": 0,
#             "object": "embedding"
#         }
#     ],
#     ...
#  }
