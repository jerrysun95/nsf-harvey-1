import numpy as np
import json, sys

def count_attributes(header):
	num_picture_attributes = 0
	num_text_attributes = 0
	
	for h in header:
		if 'PAttribute' in h:
			num_picture_attributes += 1
		elif 'TxtAttribute' in h:
			num_text_attributes += 1

	return num_picture_attributes, num_text_attributes

def read_data(data_csv):
	data = []
	num_pic_attr, num_txt_attr = count_attributes(data_csv[0].tolist())

	for l in range(1, len(data_csv)):
		line = data_csv[l]
		
		# parse each line and store information in dict
		d = {}
		d['piece_number']       = line[0]
		d['respondent_type']    = line[1]
		d['content']            = line[2]
		d['linked']             = line[3]
		d['timing']             = line[4]
		d['inside_outside']     = line[5]
		d['picture_attributes'] = []
		d['text_attributes']    = []

		pic_start = 6
		pic_range = pic_start + num_pic_attr
		for attribute in range(pic_start, pic_range):
			if line[attribute] != '':
				d['picture_attributes'].append(line[attribute])

		txt_start = pic_range
		txt_range = txt_start + num_txt_attr
		for attribute in range(txt_start, txt_range):
			if line[attribute] != '' and 'Any text' not in line[attribute]:
				d['text_attributes'].append(line[attribute])

		data.append(d)
	return data

def write_json(data, file):
	data_json = json.dumps(data, indent = 4)
	output_file_name = file[:-3] + 'json'
	with open(output_file_name, 'w') as f:
		f.write(data_json)

def main(file):
	data_csv = np.genfromtxt(file, delimiter = ',', dtype = str)
	data = read_data(data_csv)
	write_json(data, file)

if __name__ == '__main__':
	main(sys.argv[1])
