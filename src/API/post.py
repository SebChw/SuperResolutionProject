import requests
import json
import numpy as np
import cv2

path = 'data/DIV2K_valid_LR_unknown/X2/0807x2.png'
url = 'http://127.0.0.1:8000/ModelServer/super_resolute'
files = {'file': open(path, 'rb')}
ans = requests.post(url, files=files)
arr = np.asarray(json.loads(ans.json()))
cv2.imshow("img", arr)
cv2.waitKey(0)