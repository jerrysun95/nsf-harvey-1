import box, attributes, ml, json

VR_IMAGES_ID = 47340767217
R_IMAGES_ID  = 47341125866
OR_IMAGES_ID = 47341754854

VR_TEXT_ID = 47340302325
R_TEXT_ID  = 47341387936
OR_TEXT_ID = 47341790628

NOISE_ID = 47894652708

def main():
	# box.vision(NOISE_ID, 'noise_2')
	# box.frequent(.01, 'noise_2')
	# box.results('noise_2')

	# box.vision_local('C:\Users\Ben\Box Sync\Hurricane Media 2\Noise Unsynced', 'noise_3')
	# attributes.frequent(.01, 'noise_3')
	# attributes.results('noise_3')

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

	# VOTING
	# res = []
	# with open('output/optimal_respondent.json', 'r') as f:
	# 	res = json.loads(f.read())
	# for f in range(10):
	# 	min_freq = .125 + f * 0.005
	# 	print('MIN FREQ: ' + str(min_freq))
	# 	attributes.frequent(min_freq, 'vr')
	# 	attributes.frequent(min_freq, 'r')
	# 	attributes.frequent(min_freq, 'or')

	# 	attributes.results('vr')
	# 	attributes.results('r')
	# 	attributes.results('or')

	# 	o = ml.voting_classifier(['vr', 'r', 'or'], ['vr', 'r', 'or'])
	# 	res.append([min_freq, o[1]])

	# with open('output/optimal_resp.json', 'w') as f:
	# 	f.write(json.dumps(res, indent=4))

	# STACKING respondent types
	# res = []
	# for f in range(20):
	# 	min_freq = .0 + f * 0.005
	# 	print('MIN FREQ: ' + str(min_freq))
	# 	attributes.frequent(min_freq, 'vr')
	# 	attributes.frequent(min_freq, 'r')
	# 	attributes.frequent(min_freq, 'or')

	# 	attributes.results('vr')
	# 	attributes.results('r')
	# 	attributes.results('or')

	# 	o = ml.stacking_classifier(['vr', 'r', 'or'], ['vr', 'r', 'or'], 'resp', min_freq)
	# 	res.append([min_freq, o[1], o[2]])

	# with open('output/optimal_resp_svm_line.json', 'w') as f:
	# 	f.write(json.dumps(res, indent=4))

	# # STACKING signal v noise
	# res = []
	# for f in range(20):
	# 	min_freq = .0 + f * 0.005
	# 	print('MIN FREQ: ' + str(min_freq))
	# 	attributes.frequent(min_freq, 'overall')
	# 	attributes.frequent(min_freq, 'noise')

	# 	attributes.results('overall')
	# 	attributes.results('noise')

	# 	o = ml.stacking_classifier(['overall'], ['overall', 'noise'], 'noise', min_freq)
	# 	res.append([min_freq, o[1], o[2]])

	# with open('output/optimal_noise_svm_line.json', 'w') as f:
	# 	f.write(json.dumps(res, indent=4))
	run_twitter_ml(1000)

def run_twitter_ml(sample_size):
	for i in range(6):
		res = []
		for f in range(10):
			min_freq = .0 + f * 0.005
			print('MIN FREQ: ' + str(min_freq))
			attributes.frequent(min_freq, 'tweets_random_sample_gv_' + str(i) + '_' + str(sample_size))
			attributes.frequent(min_freq, 'overall')

			attributes.results('tweets_random_sample_gv_' + str(i) + '_' + str(sample_size))
			attributes.results('overall')

			o = ml.stacking_classifier(['overall'], ['overall', 'tweets_random_sample_gv_' + str(i) + '_' + str(sample_size)], 
				'tweets_random_sample_gv_' + str(i) + '_' + str(sample_size), min_freq, save=True)

		with open('tweets/sample_results_' + str(i) + '_' + str(sample_size) + '.json', 'w') as f:
			f.write(json.dumps(res, indent=4))

if __name__ == '__main__':
	main()
