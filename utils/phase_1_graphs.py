import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json

# Computer vision frequency analysis
def freq_analysis_bar():
	sns.set(style='white')

	x = ('0.00', '0.05', '0.10', '0.15', '0.20', '0.25')
	n = np.genfromtxt('../analysis/frequent.csv')

	d = {'minimum frequency':[], 'attributes':[], 'Type': []}
	for i in range(1, len(n)):
		d['minimum frequency'].append(x[i])
		d['minimum frequency'].append(x[i])
		d['minimum frequency'].append(x[i])

		d['attributes'].append(n[i][2]/n[0][2])
		d['attributes'].append(n[i][1]/n[0][1])
		d['attributes'].append(n[i][0]/n[0][0])

		d['Type'].append('Official Rescuers')
		d['Type'].append('Rescuees')
		d['Type'].append('Volunteer Rescuers')

	df = pd.DataFrame(data=d)
	# print(df)
	ax = sns.barplot(x='minimum frequency', y='attributes', hue='Type', data = df, palette='Blues')

	sns.despine(bottom=True, left=True)
	plt.xlabel('Minimum Frequency')
	plt.ylabel('Percentage of Total Attributes')
	plt.title('Computer Vision Attributes Frequency Analysis')
	plt.show()

def freq_analysis_violin():
	sns.set(style='white')

	with open('../output/vr.json') as f:
			vr = json.loads(f.read())

	with open('../output/r.json') as f:
			r = json.loads(f.read())

	with open('../output/or.json') as f:
			o = json.loads(f.read())

	allf = {}
	vf = {}
	for x in vr:
		for a in x['picture_attributes']:
			if a not in allf:
				allf[a] = 0
			if a not in vf:
				vf[a] = 0
			allf[a] += 1
			vf[a] += 1

	rf = {}
	for x in r:
		for a in x['picture_attributes']:
			if a not in allf:
				allf[a] = 0
			if a not in rf:
				rf[a] = 0
			allf[a] += 1
			rf[a] += 1

	of = {}
	for x in o:
		for a in x['picture_attributes']:
			if a not in allf:
				allf[a] = 0
			if a not in of:
				of[a] = 0
			allf[a] += 1
			of[a] += 1

	for x in allf:
		allf[x] /= float(len(vr) + len(r) + len(o))

	for x in vf:
		vf[x] /= float(len(vr))

	for x in rf:
		rf[x] /= float(len(r))

	for x in of:
		of[x] /= float(len(o))


	d = {'Frequency':[], 'Type':[]}
	for x in vf:
		d['Frequency'].append(vf[x])
		d['Type'].append('Volunteer Resucers')

	for x in rf:
		d['Frequency'].append(rf[x])
		d['Type'].append('Rescuees')

	for x in of:
		d['Frequency'].append(of[x])
		d['Type'].append('Official Rescuers')

	for x in allf:
		d['Frequency'].append(allf[x])
		d['Type'].append('Combined')

	df = pd.DataFrame(data=d)
	ax = sns.violinplot(x='Type', y='Frequency', data=df, palette='Blues')
	plt.title('Computer Vision Attributes Frequency Analysis')

	sns.despine(bottom=True, left=True)
	plt.show()

freq_analysis_violin()