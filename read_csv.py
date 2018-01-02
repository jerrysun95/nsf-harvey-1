import numpy as np
import pandas as pd
import json, sys

# Reads in an excel file and writes content to csv file
def excel_to_csv(file, sheet='Sheet1'):
    data_xls = pd.read_excel(file, sheet, index_col=None)
    csv_name = '%s%s' % (file[:-4], 'csv')
    data_xls.to_csv(csv_name, encoding='utf-8', index=False)

# Reads in an excel file and returns a pandas dataframe
def read_excel(file, sheet='Sheet1'):
    data_xls = pd.read_excel(file, sheet, index_col=None, keep_default_na=False)
    return data_xls

# Counts and returns the number of picture attributes and text attributes from headers
def count_attributes(header):
    num_picture_attributes = 0
    num_text_attributes = 0
    
    for h in header:
        if 'PAttribute' in h:
            num_picture_attributes += 1
        elif 'TxtAttribute' in h:
            num_text_attributes += 1

    return num_picture_attributes, num_text_attributes

# Reads and parses a csv formatted dataset
def read_data(data_csv, headers):
    data = []
    num_pic_attr, num_txt_attr = count_attributes(headers)

    for l in range(len(data_csv)):
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
        for attributes in range(pic_start, pic_range):
            if line[attributes] != '':
                d['picture_attributes'].append(line[attributes])

        txt_start = pic_range
        txt_range = min(txt_start + num_txt_attr, len(line))
        for attribute in range(txt_start, txt_range):
            if line[attribute] != '' and 'Any text' not in line[attribute]:
                d['text_attributes'].append(line[attribute])

        data.append(d)
    return data

# Writes json compatible dataset to file
# If file exists, appends data to end
def write_json(image_data, file='output/human.json'):
    json_data = None
    with open(file, 'a+') as output_file:
        # read in existing data if present
        data = output_file.read()
        try:
            json_data = json.loads(data)
        except:
            json_data = []

        # append new image to json and write back to file
        json_data += image_data
        json_data = json.dumps(json_data, indent = 4)
        output_file.seek(0)
        output_file.write(json_data)

# Parse excel or csv from file
# Writes/appends output to output/human.json
def parse_from_file(file):
    if '.xlsx' in file:
        excel_to_csv(file)
        file = file[:-4] + 'csv'
    data_csv = np.genfromtxt(file, delimiter=',', dtype=str)
    headers = data_csv[0]
    data_csv = np.delete(data_csv, 0, 0)
    data = read_data(data_csv, headers)
    write_json(data)

# Parse excel or csv from data
# Writes/appends output to output/human.json
def parse_from_data(values, headers):
    data = read_data(values, headers)
    write_json(data)

def main(file):
    parse(file)

if __name__ == '__main__':
    main(sys.argv[1])
