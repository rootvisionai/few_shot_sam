import torch
from flask import Flask, request, jsonify
import requests
from waitress import serve
import numpy as np
import os
import sys
import cv2
import json
import queue
import time
import threading

import server_utils as utils

cfg = utils.load_config("./config.yml")
app = Flask(__name__)
ExtractInQueue = queue.Queue(maxsize=10)
ExtractOutDict = {}
GenerateInQueue = queue.Queue(maxsize=4)
GenerateOutDict = {}
base = "localhost:8081"
class Forwarder:
    def __init__(self, in_queue, out_dict, freq):
        self.freq = freq
        self.in_queue = in_queue
        self.out_dict = out_dict

    def run(self):
        while True:
            if not self.in_queue.empty() and len(self.out_dict) < 4:
                data = self.in_queue.get()
                response = requests.post(
                    data["url"],
                    data=json.dumps(data["data"]),
                    headers={"Content-Type": "application/json"}
                )
                self.out_dict[data["timestamp"]] = {"data": response.text, "url": data["url"]}
@app.route('/health', methods=['GET'])
def health_check():
    return json.dumps({"status": "active"})

@app.route('/forwarder/extract_features', methods=['POST'])
def extract():

    timestamp = time.time()
    if not ExtractInQueue.full():
        ExtractInQueue.put({
            "url": f"http://{base}/extract_features",
            "data": request.json,
            "timestamp": timestamp
        })
    else:
        return json.dumps({"warning": "Queue is full, please wait and try again."})

    while True:
        if timestamp in ExtractOutDict:
            response = ExtractOutDict[timestamp]["data"]
            del ExtractOutDict[timestamp]
            break

    return response

@app.route('/forwarder/generate/<gen_type>', methods=['POST'])
def generate(gen_type):

    timestamp = time.time()
    if not GenerateInQueue.full():
        GenerateInQueue.put({
            "url": f"http://{base}/generate/{gen_type}",
            "data": request.json,
            "timestamp": timestamp
        })
    else:
        return json.dumps({"warning": "Queue is full, please wait and try again."})

    while True:
        if timestamp in GenerateOutDict:
            response = GenerateOutDict[timestamp]["data"]
            del GenerateOutDict[timestamp]
            break

    return response

def run_forwarder(forwarder):
    forwarder.run()

if __name__ == '__main__':

    forwarder_extract = Forwarder(
        in_queue=ExtractInQueue,
        out_dict=ExtractOutDict,
        freq=10
    )

    forwarder_generate = Forwarder(
        in_queue=GenerateInQueue,
        out_dict=GenerateOutDict,
        freq=10
    )

    fps = []
    fps.append(threading.Thread(target=run_forwarder, args=(forwarder_extract,)))
    fps.append(threading.Thread(target=run_forwarder, args=(forwarder_generate,)))

    for fp in fps:
        fp.daemon = True
        fp.start()

    # app.run(host="0.0.0.0", port=8080, debug=False)
    serve(app, host="0.0.0.0", port=8080)