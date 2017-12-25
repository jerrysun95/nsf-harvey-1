import json

def load_json():
	with open('output/human.json', 'r') as h, open('output/vision.json', 'r') as v:		
		human = json.loads(h.read())
		vision = json.loads(v.read())

		return human, vision

def combine_json(human, vision):
	combined = {}

	for h in human:
		piece = h['piece_number']
		if piece not in combined:
			combined[piece] = {}
		combined[piece]['human']  = h['picture_attributes']

	for v in vision:
		piece = v['piece_number']
		if piece not in combined:
			combined[piece] = {}
		combined[v['piece_number']]['vision'] = v['picture_attributes']

	return combined

def compare_json(combined):
	result = {}
	for key, piece in combined.iteritems():
		h = set()
		v = set()
		if 'human' in piece:
			h = set(piece['human'])

		if 'vision' in piece:
			v = set(piece['vision'])

		result[key] = {}
		result[key]['intersect']   = list(h & v)
		result[key]['only_human']  = list(h - v)
		result[key]['only_vision'] = list(v - h)
	return result

def compare():
	human, vision = load_json()
	combined = combine_json(human, vision)
	result = compare_json(combined)

	with open('output/compare.json', 'w') as output:
		output.write(json.dumps(result, indent=4, sort_keys=True))

def main():
	compare()

if __name__ == '__main__':
	main()