import box, compare, read_csv
import google_vision as gv 

'''
Vision:
	1. Get (access to) all pictures from box
	2. Send all pictures through google vision
	3. Aggregate results into vision.json
Human:
	1. Get all excel sheets from box
	2. Parse excel sheets (through read_csv)
	3. Aggregate results into human.json
After both vision and human have finished, run compare.py to generate output.json 
'''
# Box folder id
respondent_folder_id = 0

# Sends all images in images folder to google vision
def send_images(images)
	entries = images['entries']
	for image in entries:
		image_id = image['id']
		image_name = image['name']
		box.send_to_vision(image_name, image_id)

# Sends all the images of a respondent type (rescuee, etc) to google_vision
def send_respondent(respondent_type):
	respondents = respondent_type['entries']
	for respondent in respondents:
		if respondent['type'] == 'folder':
			folder_id = respondent['id']

			# contents of a specific respondent
			data = box.items(folder_id)
			e = data['entries']

			# find images folder if exists
			for item in e:
				if item['name'] == 'images':
					images_folder_id = item['id']
					images = box.items(images_folder_id)
					get_images(images)
					break

# Starts at Respondent_Data folder in box and sends all images for each type to google_vision
def send_respondents():
	respondent_data = box.items(respondent_folder_id)
	respondent_types = respondent_data['entries']

	# iterate over each respondent type (rescuee, etc)
	for respondent_type in respondent_types:
		if respondent_type['type'] == 'folder':
			folder_id = respondent_type['id']

			# contents of a respondent type folder
			respondent_type_data = box.items(folder_id)
			send_respondent(respondent_type_data)


def main():

	'''
	Revised main.py control flow:
	1. Iterate through box directories and send to appropriate process:
		Respondent_Data/
			Rescuee Folder/
				Rescuee1/
					images/		** send these images through google vision when we get here
						*.jpg
				...
				R_*_codes.xlsx 	** send these through read_csv when we get here
			Official Rescuer Folder/
			Volunteer Rescuer Folder/
	2. Compare results
		Step 1 automatically outputs
		compare.py just needs to be run (it will pull from human.json and vision.json and produce output.json)

	Notes:
		read_csv.py and send_to_vision() always append (this means that if you run them on the same picture twice, the output will show up twice in json)
		this is hard to test because box is set up to our developer account which doesn't have any of the data
		I added a method (box.items(folder_id)) in box to view folder contents in box that I use to walk the file system. Looks something like the following:
			get folder_id
			call box.items(folder_id)
			this returns a box response object
			get entries using response['entries']
			entries is a dict with keys for 'id' (folder_id or file_id), 'name' (ex: obama.jpeg), and 'type' (folder or file)
		I was planning on hard coding the folder ids for the top level respondent type folders (Rescuee Folder, etc) to avoid some box network calls
		While we still have a small sample size, I think it is okay if we just erase human/vision/output.json each time and just always grab all the contents. We can get smarter about it later to only get new files
	'''

if __name__ == '__main__':
	main()