import json
import pylab as plt
import plotly.plotly as py
import numpy as np

vr_attributes = None
with open('output/vr_attributes.json', 'r') as f:
	vr_attributes = json.loads(f.read())

r_attributes = None
with open('output/r_attributes.json', 'r') as f:
	r_attributes = json.loads(f.read())

s = set()
for attribute in r_attributes:
	s.add(attribute['name'])

for attribute in vr_attributes:
	s.add(attribute['name'])

data = {}

for attribute in s:
	data[attribute] = {'vr':0, 'r':0}

for d in r_attributes:
	data[d['name']]['r'] = d['freq']

for d in vr_attributes:
	data[d['name']]['vr'] = d['freq'] 

result = []
for k, v in data.iteritems():
	result.append({'name':k, 'r':v['r'], 'vr':v['vr']})
result = sorted(result, key=lambda x:(x['r'] - x['vr']))

fig = plt.figure()

index = np.arange(len(result))
bar_width = .2

x = [x['r'] for x in result]
y = [y['vr'] for y in result]
names = [n['name'] for n in result]


r_bar = plt.bar(index, x, bar_width, label='Rescuees')
vr_bar = plt.bar(index + bar_width, y, bar_width, label='Volunter Rescuers')

plt.xlabel('Attributes')
plt.ylabel('Frequency')
plt.title('Frequent Attributes Across Respondent Types')
plt.xticks(index + bar_width / 2, names, rotation=90)
plt.legend()
plt.tight_layout()

plt.show()
