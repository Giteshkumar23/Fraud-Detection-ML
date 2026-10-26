
import pandas as pd
import json
import urllib.request
import urllib.error

# Load one transaction from the dataset
df = pd.read_csv("data/creditcard.csv", nrows=2)
transaction = df.drop(columns=["Class"]).iloc[0].to_dict()

# Send transaction to the local API
url = "http://127.0.0.1:5000/predict"
body = json.dumps(transaction).encode("utf-8")

request = urllib.request.Request(
    url,
    data=body,
    headers={"Content-Type": "application/json"},
    method="POST"
)

try:
    with urllib.request.urlopen(request) as response:
        result = json.loads(response.read().decode("utf-8"))
        print("API test successful!")
        print(json.dumps(result, indent=4))
except urllib.error.HTTPError as error:
    print("API returned an error:", error.code)
    print(error.read().decode("utf-8"))
except Exception as error:
    print("Could not connect to the API:", error)
