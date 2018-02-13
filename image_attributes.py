import json

def count_attributes(t):
	with open('output/' + t + '.json', 'r') as f:
		data = json.loads(f.read())
		print('Type: ' + t)
		print('Number of images: ' + str(len(data)))
		print('')

		attributes = {}
		for d in data:
			for i in range(len(d['picture_attributes'])):
				a = d['picture_attributes'][i]
				if a not in attributes:
					attributes[a] = []
				attributes[a].append(d['picture_attributes_scores'][i])

		return attributes, len(data)

def print_freq_attributes(min_freq, num_images, attributes, t, file):
	f = []
	for a in attributes.keys():
		freq = round(float(len(attributes[a])) / num_images, 2)
		avg_conf = round(sum(attributes[a]) / len(attributes[a]), 2)
		if freq >= min_freq:
			f.append({'name':a, 'freq':freq, 'avg_conf':avg_conf})

	print(t)
	print('Minimum Frequency: ' + str(min_freq))
	print('Total Attributes: ' + str(len(attributes)))
	print('Frequent Attributes: ' + str(len(f)))
	s = sorted(f, key=lambda x:x['freq'], reverse=True)
	for x in s:
		print('	' + x['name'] + ': \n\t\tfreq: ' + str(x['freq']) + '\n\t\tconf: ' + str(x['avg_conf']))
	print('')
	with open('output/' + file, 'w') as f:
		f.write(json.dumps(s, indent=4))

vr, num_images = count_attributes('vr')
print_freq_attributes(.15, num_images, vr, 'Volunteer Rescuers', 'vr_attributes.json')

r, num_images = count_attributes('r')
print_freq_attributes(.15, num_images, r, 'Rescuers', 'r_attributes.json')

o, num_images = count_attributes('or')
print_freq_attributes(.15, num_images, o, 'Official Rescuers', 'or_attributes.json')