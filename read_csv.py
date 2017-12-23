import numpy as np
import pandas as pd
import json, sys

def excel_to_csv(file, sheet='Sheet1'):
    # wb = xlrd.open_workbook(file)
    # sh = wb.sheet_by_name(sheet)
    # csv_name = 'materials/%s%s' % (file[:-4], 'csv')

    # with open(csv_name, 'w') as output:
    #     wr = csv.writer(output, quoting = csv.QUOTE_ALL)
    #     for r in range(sh.nrows):
    #         wr.writerow(sh.row_values(r))
    data_xls = pd.read_excel(file, sheet, index_col=None)
    csv_name = '%s%s' % (file[:-4], 'csv')
    data_xls.to_csv(csv_name, encoding='utf-8', index=False)

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

def csv_to_json(file):
    data_csv = np.genfromtxt(file, delimiter=',', dtype = str)
    data = read_data(data_csv)
    write_json(data)

def main(file):
    if '.xlsx' in file:
        excel_to_csv(file)
        file = file[:-4] + 'csv'
    csv_to_json(file)

if __name__ == '__main__':
    main(sys.argv[1])
