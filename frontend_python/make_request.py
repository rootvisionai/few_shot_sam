import requests
import json
import interface_utils as utils
import time, os

# Define the URL of the endpoint: http://fewshotsam.rootvisionai.net
url = "http://fewshotsam.rootvisionai.net/forwarder/extract_features"  # replace with your actual endpoint

# make get request
response = os.system("curl --request GET http://localhost:8080/health")
print(response)

# Convert json to dict
with open("request_content.json", "r") as fp:
    data_json = json.load(fp)

# Set the headers for the request
headers = {
    "Content-Type": "application/json"
}

# Send the POST request
t0 = time.time()
response = requests.post(url, data=json.dumps(data_json), headers=headers)
t1 = time.time()
data_json = json.loads(response.text)
image_path = "../support_images/test.jpg"
data_json["image_path"] = image_path
init_image = utils.import_image(image_path)
data_json["image"] = utils.numpy_to_base64(init_image)

# Print the response
try:
    print(data_json["error"] if "error" in data_json.keys() else data_json.keys())
except:
    print(data_json)

# Define the URL of the endpoint
url = "http://fewshotsam.rootvisionai.net/forwarder/generate/all"

# Send the POST request
t1 = time.time()
response = requests.post(url, data=json.dumps(data_json), headers=headers)
t2 = time.time()
data_json = json.loads(response.text)

# Print the response
try:
    print(data_json["error"] if "error" in data_json.keys() else data_json.keys())
except:
    print(data_json)

# save annotation
with open("test.json", "w") as fp:
    json.dump(data_json["coco_json"], fp, indent=4)

print(f"Request.1 Time: {t1-t0} | Request.2 Time: {t2-t1}")

