import box, attributes, ml, json

VR_IMAGES_ID = 47340767217
R_IMAGES_ID  = 47341125866
OR_IMAGES_ID = 47341754854

VR_TEXT_ID = 47340302325
R_TEXT_ID  = 47341387936
OR_TEXT_ID = 47341790628

def main():
	# # Send images through computer vision
	# print('Sending images through computer vision')
	# box.vision(VR_IMAGES_ID, 'vr')
	# box.vision(R_IMAGES_ID,  'r')
	# box.vision(OR_IMAGES_ID, 'or')
	# box.vision_text(VR_TEXT_ID, 'vr_text')
	# box.vision_text(R_TEXT_ID,  'r_text')
	# box.vision_text(OR_TEXT_ID, 'or_text')
	# print('Finished\n')

	# # Frequency analysis of computer vision results
	# print('Creating frequent attributes lists')
	# min_freq = .08
	# attributes.frequent(min_freq, 'vr')
	# attributes.frequent(min_freq, 'r')
	# attributes.frequent(min_freq, 'or')
	# attributes.frequent(min_freq, 'vr_text')
	# attributes.frequent(min_freq, 'r_text')
	# attributes.frequent(min_freq, 'or_text')
	# print('Finished')

	# # Process computer vision results with frequency results
	# print('Processing results and trimming image attributes')
	# attributes.results('vr')
	# attributes.results('r')
	# attributes.results('or')
	# attributes.results('vr_text')
	# attributes.results('r_text')
	# attributes.results('or_text')
	# print('Finished\n')

	# # Create predictive model from supervised learning
	# print('Training predictive model')
	# ml.aggregate_freq(['vr', 'r', 'or'])
	# ml.iter_results('vr', ml.VR)
	# ml.iter_results('r',  ml.R)
	# ml.iter_results('or', ml.OR)
	# ml.partition_data()
	# print('svm: ' + str(ml.svm()))
	# print('naive bayes: ' + str(ml.naive_bayes()))
	# print('nearest neighbors: ' + str(ml.nearest_neighbor(5)))
	# print('Finished\n')

	res = []
	with open('output/optimal_respondent.json', 'r') as f:
		res = json.loads(f.read())
	for f in range(10):
		min_freq = .125 + f * 0.005
		print('MIN FREQ: ' + str(min_freq))
		attributes.frequent(min_freq, 'vr')
		attributes.frequent(min_freq, 'r')
		attributes.frequent(min_freq, 'or')

		attributes.results('vr')
		attributes.results('r')
		attributes.results('or')

		o = ml.voting_classifier(['vr', 'r', 'or'], ['vr', 'r', 'or'])
		res.append([min_freq, o[1]])

	with open('output/optimal_resp.json', 'w') as f:
		f.write(json.dumps(res, indent=4))

if __name__ == '__main__':
	main()
