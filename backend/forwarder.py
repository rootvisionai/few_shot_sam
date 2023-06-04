import json
import requests


class Forwarder:
    def __init__(self, in_queue, out_dict, freq):
        self.freq = freq
        self.in_queue = in_queue
        self.out_dict = out_dict

    def run(self):
        while True:
            if not self.in_queue.empty() and len(self.out_dict) < 4:
                data = self.in_queue.get()
                print(f"FORWARDING TO {data['url']}")
                response = requests.post(
                    data["url"],
                    data=json.dumps(data["data"]),
                    headers={"Content-Type": "application/json"}
                )
                self.out_dict[data["timestamp"]] = {"data": response.text, "url": data["url"]}