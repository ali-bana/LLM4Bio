import json

d = {}
d[2] = {2: 3, 3: 4}
d[3] = {5: 3, 4: 4}

with open('temp.json', 'w') as f:
    json.dump(d, f)
