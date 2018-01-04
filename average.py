import json

comp = None
with open('output/compare.json', 'r') as f:
	comp = json.loads(f.read())

vis = None
with open('output/vision.json', 'r') as f:
	vis = json.loads(f.read())

def get_name(n):
	i1 = n.find('_') + 1
	i2 = n.rfind('_')
	return n[i1:i2]

values = {}
num_pics = {}
for key, attributes in comp.iteritems():
	name = get_name(key)
	if name not in values:
		values[name] = {}
		num_pics[name] = 0
	num_pics[name] += 1
	for a in (attributes['intersect'] + attributes['only_vision']):
		if a not in values[name]:
			values[name][a] = []
		values[name][a].append(vis[key][a])
with open('output/values.json', 'w') as f:
	f.write(json.dumps(values, indent=4))

averages = {}
for name, attributes in values.iteritems():
	temp = []
	for a, vals in attributes.iteritems():
		# avg = sum(vals) / num_pics[name]
		avg = sum(vals) / len(vals)
		temp.append([a, avg])
	averages[name] = sorted(temp, key=lambda x:x[1], reverse=True)

with open('output/averages.json', 'w') as f:
	f.write(json.dumps(averages, indent=4, sort_keys=True))