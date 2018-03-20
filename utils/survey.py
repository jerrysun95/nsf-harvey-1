import json
import pylab as plt
import matplotlib.pyplot as py

with open('../output_old/survey.csv', 'r') as f:
	survey = f.readlines()

num_responses = len(survey) - 1

with open('../output_old/combined_results.json') as f:
	frequent = json.loads(f.read())

with open('../output_old/combined.json') as f:
	attributes = json.loads(f.read())

sresults = {}
aresults = {}
names = []

def get_results():
	header = survey[0].replace('"', '').replace('\n', '').lower().split(',')
	for n in range(1, len(header)):
		name = header[n]
		sresults[name] = {}
		aresults[name] = {}
		names.append(name)

	for d in frequent:
		name = d['name']
		if name in aresults:
			for a in d['attributes']:
				sresults[name][a] = 0
				aresults[name][a] = 0

	for d in attributes:
		name = d['piece_name']
		if name in names:
			for i in range(0, len(d['picture_attributes'])):
				attribute = d['picture_attributes'][i]
				if attribute in aresults[name]:
					aresults[name][attribute] = d['picture_attributes_scores'][i]

	for l in range(1, len(survey)):
		line = survey[l].replace('"', '').replace('\n', '').split(',')
		for c in range(1, len(line)):
			choices = line[c].split(';')
			name = names[c-1]
			for choice in choices:
				if choice in sresults[name]:
					sresults[name][choice] += (1. / num_responses)

	with open('sresults.json', 'w') as f:
		f.write(json.dumps(sresults, indent=4))

	with open('aresults.json', 'w') as f:
		f.write(json.dumps(aresults, indent=4))

def graph_results_scatter():
	x = []
	y = []

	for pic in sresults:
		for attribute in sresults[pic]:
			x.append(sresults[pic][attribute])
			y.append(aresults[pic][attribute])

	plt.scatter(x, y)
	plt.xlabel('Survey Confidence Level')
	plt.ylabel('Google Vision Confidence Level')
	plt.title('Machine Attributes Survey Results')

	plt.show()

def graph_results_line():
	x = []
	y = []

	d = float(num_responses)

	for i in range(0, num_responses + 1):
		# val = num_responses * 1.0
		x.append(i / d)
		y.append([])

	for pic in sresults:
		for attribute in sresults[pic]:
			index = int(sresults[pic][attribute] * 4)
			y[index].append(aresults[pic][attribute])

	plt.plot(x, [(sum(v) / len(v)) for v in y])
	plt.xlabel('Survey Confidence Level')
	plt.ylabel('Google Vision Confidence Level')
	plt.title('Machine Attributes Survey Results')
	plt.xticks(x)
	plt.yticks([y/10. for y in range(0, 11)])

	plt.show()

get_results()
graph_results_line()
# graph_results_scatter()