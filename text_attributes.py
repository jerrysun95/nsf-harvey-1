import json

def count_attributes(t):
	with open('output/' + t + '.json.txt', 'r') as f:
		data = json.loads(f.read())
		print('Type: ' + t)
		print('Number of images: ' + str(len(data)))
		print('')

		attributes = {}
		for d in data:
			for i in range(len(d['words'])):
				a = d['words'][i]
				if a not in attributes:
					attributes[a] = []
				attributes[a].append(1)

		return attributes, len(data)

def print_freq_attributes(min_freq, num_images, attributes, t, file):
	f = []
	for a in attributes.keys():
		freq = round(float(len(attributes[a])) / num_images, 2)
		if freq >= min_freq:
			f.append({'name':a, 'freq':freq})

	print(t)
	print('Minimum Frequency: ' + str(min_freq))
	print('Total Attributes: ' + str(len(attributes)))
	print('Frequent Attributes: ' + str(len(f)))
	s = sorted(f, key=lambda x:x['freq'], reverse=True)
	for x in s:
		print('	' + x['name'] + ': \n\t\tfreq: ' + str(x['freq']))
	print('')
	with open('output/' + file, 'w') as f:
		f.write(json.dumps(s, indent=4))

txt, num_images = count_attributes('text')
print_freq_attributes(.15, num_images, txt, 'text', 'txt_attributes.json')

# r, num_images = count_attributes('r')
# print_freq_attributes(.15, num_images, r, 'Rescuers', 'r_attributes.json')

# o, num_images = count_attributes('or')
# print_freq_attributes(.15, num_images, o, 'Official Rescuers', 'or_attributes.json')