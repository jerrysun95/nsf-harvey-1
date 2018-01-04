import json

comp = None
with open('output/compare.json', 'r') as f:
	comp = json.loads(f.read())

vis = None
with open('output/vision.json', 'r') as f:
	vis = json.loads(f.read())

def get_name(n):
	i2 = n.rfind('_')
	return n[:i2]

values = {}
num_pics = {}
attr = {}
for key, attributes in comp.iteritems():
	name = get_name(key)
	if name not in values:
		values[name] = {}
		num_pics[name] = 0
	num_pics[name] += 1
	for a in (attributes['intersect'] + attributes['only_vision']):
		if a not in values[name]:
			values[name][a] = []
		if a not in attr:
			attr[a] = 0
		values[name][a].append(vis[key][a])
		attr[a] += 1
with open('output/values.json', 'w') as f:
	f.write(json.dumps(values, indent=4))

averages = {}
for name, attributes in values.iteritems():
	temp = []
	for a, vals in attributes.iteritems():
		# avg = sum(vals) / num_pics[name]
		avg = (sum(vals) / len(vals)) + .1
		adj = (avg * avg) - .1

		temp.append([a, adj])
	averages[name] = sorted(temp, key=lambda x:x[1], reverse=True)

for name in sorted(averages.keys()):
	attributes = averages[name]
	print('Name: ' + name)
	print('High: ' + str(attributes[0][1]))
	print('Low:  ' + str(attributes[len(attributes)-1][1]))
	print('')

def frequency_counts(min_freq):
	frequent = []
	for a in attr.keys():
		if attr[a] >= min_freq:
			frequent.append(a)

	print('Number of Attributes: ' + str(len(attr)))
	print('Number of Attributes (freq >= ' + str(min_freq) + '): ' + str(len(frequent)))
	print('Attributes over 10:')
	for a in frequent:
		print(a)

with open('output/averages.json', 'w') as f:
	f.write(json.dumps(averages, indent=4, sort_keys=True))

with open('output/attributes.json', 'w') as f:
	l = []
	for a in attr.keys():
		l.append({'name':a, 'frequency':attr[a]})
	
	frequency_counts(15)
	
	l = sorted(l, key=lambda x:x['frequency'], reverse=True)
	f.write(json.dumps(l, indent=4))