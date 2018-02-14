import box, read_csv
import google_vision as gv
import json

MEDIA_FOLDER_ID = 44111513325

VR_FOLDER_ID = 44087771250
R_FOLDER_ID  = 44087753842
OR_FOLDER_ID = 44087780508

VR_DEST_ID = 44224186584
R_DEST_ID  = 44224137472
OR_DEST_ID = 44224175952

TARGET_FOLDER_ID = 44208502416

error = []
def create_vision_output(vision_data):
	vout = {}
	for v in vision_data:
		name = v['piece_number']
		vout[name] = {}
		vout[name]['name'] = v['piece_name']
		for x in range(len(v['picture_attributes'])):
			vout[name][v['picture_attributes'][x]] = v['picture_attributes_scores'][x]

	with open('vision.json', 'a+') as f:
		f.write(json.dumps(vout, indent=4))

def copy_images(entries, dest_folder_id):
	for entry in entries:
		if entry['type'] == 'folder':
			if entry['name'].lower() != 'text' and entry['name'].lower() != 'videos':
				response = box.items(entry['id'])
				copy_images(response['entries'], dest_folder_id)
		else:
			name = entry['name'].lower()
			if '.jpg' in name or '.jpeg' in name or '.png' in name:
				try:
					box.copy(entry['id'], dest_folder_id)
				except:
					error.append(entry['id'])

def images(src_folder_id, dest_folder_id):
	response = box.items(src_folder_id)
	entries = response['entries']
	copy_images(entries, dest_folder_id)

def vision(src_folder_id, output_file):
	vision_data = []
	response = box.items(src_folder_id)
	entries = response['entries']

	for entry in entries:
		name = entry['name'].lower()
		if entry['type'] == 'file' and '.jpg' in name or '.png' in name or '.jpeg' in name:
			vision_data.append(box.send_to_vision(name, entry['id']))

	# with open(output_file, 'w') as f:
	# 	f.write(json.dumps(vision_data, indent=4))

def main():
	# response = box.items(MEDIA_FOLDER_ID)
	# entries = response['entries']
	
	# vision_data = []
	# human_data = []
	# for entry in entries:
	# 	name = entry['name'].lower()
	# 	if '.jpg' in name or '.png' in name or '.jpeg' in name:
	# 		vision_data.append(box.send_to_vision(name, entry['id']))
	# 	elif '.xlsx' in name or '.csv' in name:
	# 		human_data += box.parse_excel(entry['id'])

	# # gv.output_to_file(json_data)

	# print('Vision Results: ' + str(len(vision_data)))
	# print('Human Results: ' + str(len(human_data)))
	# compare.compare(human_data, vision_data)

	# images(VR_FOLDER_ID, VR_DEST_ID)
	# images(R_FOLDER_ID, R_DEST_ID)
	# images(OR_FOLDER_ID, OR_DEST_ID)


	# vision(VR_DEST_ID, 'output/vr.json')
	# vision(R_DEST_ID, 'output/r.json')
	vision(OR_DEST_ID, 'output/or.json')

if __name__ == '__main__':
	main()
