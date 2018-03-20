import json, sys

FREQ = sys.argv[1]

# Frequency analysis of attributes
# Writes list of sorted frequencies above MIN_FREQ to output file
def frequent(min_freq, t):
	# Read in file
	with open('output/' + t + '.json', 'r') as f:
		data = json.loads(f.read())

	# Count attributes in data
	attributes = {}
	for d in data:
		for a in d['attributes']:
			if a not in attributes:
				attributes[a] = 0
			attributes[a] += 1	

	# Check frequencies of attributes
	freqs = []
	for a in attributes:
		freq = round(float(attributes[a]) / len(data), 2)
		if freq >= min_freq:
			freqs.append({'name':a, 'freq':freq})

	# Sort by frequency
	s = sorted(freqs, key=lambda x:x['freq'], reverse=True)

	# Output to file
	with open('output/' + t + '_attributes.json', 'w') as f:
		f.write(json.dumps(s, indent=4))

# Process computer vision results by trimming non-frequent attributes
def results(t):
	# Create list of frequent attributes	
	with open('output/' + t + '_attributes.json') as f:
		freq = json.loads(f.read())
		freq = [x['name'] for x in freq]

	# Read in computer vision output
	with open('output/' + t + '.json') as f:
		data = json.loads(f.read())

	# Trim non-frequent attributes
	res = []
	for x in data:
		d = {}
		d['name'] = x['name']
		d['attributes'] = []
		for a in x['attributes']:
			if a in freq:
				d['attributes'].append(a)
		res.append(d)

	# Write results to output file
	with open('output/' + t + '_results.json', 'w') as f:
		f.write(json.dumps(res, indent=4))

# Combines the results of all files in files and writes to output file denoted by t
def combine(files, t):
	data = []
	for file in files:
		with open('output/' + file + '.json') as f:
			data += json.loads(f.read())

	with open('output/' + t + '.json') as f:
		f.write(json.dumps(data, indent=4))