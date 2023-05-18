import requests
import json

# Define the URL of the endpoint
url = "http://localhost:5000/extract_features"  # replace with your actual endpoint

# Convert json to dict
with open("request_content.json", "r") as fp:
    data_json = json.load(fp)

# Set the headers for the request
headers = {
    "Content-Type": "application/json"
}

# Send the POST request
response = requests.post(url, data=json.dumps(data_json), headers=headers)

# Print the response
print(response.text)
