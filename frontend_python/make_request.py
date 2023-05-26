import requests
import json
import interface_utils as utils
import time

# Define the URL of the endpoint: http://fewshotsam.rootvisionai.net
url = "http://fewshotsam.rootvisionai.net/extract_features"  # replace with your actual endpoint

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
image_path = "../test.jpg"
data_json["image_path"] = image_path
init_image = utils.import_image(image_path)
data_json["image"] = utils.numpy_to_base64(init_image)

# Print the response
print(data_json)

# Define the URL of the endpoint
url = "http://fewshotsam.rootvisionai.net/generate/all"

# Send the POST request
response = requests.post(url, data=json.dumps(data_json), headers=headers)
t2 = time.time()
data_json = json.loads(response.text)
with open("test.json", "w") as fp:
    json.dump(data_json["coco_json"], fp, indent=4)

print(data_json)
print(f"Request.1 Time: {t1-t0} | Request.2 Time: {t2-t1}")

