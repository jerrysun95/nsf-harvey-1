import json

def get_attributes(file):
	with open(file, 'r') as f:
		data = json.loads(f.read())

		attributes = set()
		for d in data:
			attributes.add(d['name'])

		return attributes

def get_vision_data(file):
	with open(file, 'r') as f:
		return json.loads(f.read())

def compare_results(frequent_set, image_data):
	attributes = image_data['picture_attributes']
	matches = []
	extra = []
	for a in attributes:
		if a in frequent_set:
			matches.append(a)
		else:
			extra.append(a)
	return matches, extra

def compare(t):
	attributes = get_attributes('output/' + t + '_attributes.json')
	data = get_vision_data('output/' + t + '.json')
	results = []
	for d in data:
		matches, extra = compare_results(attributes, d)
		r = {'name':d['piece_name'], 'extra':extra, 'matches':matches}
		results.append(r)

	with open('output/' + t + '_results.json', 'w') as f:
		f.write(json.dumps(results, indent=4))

compare('vr')
compare('r')