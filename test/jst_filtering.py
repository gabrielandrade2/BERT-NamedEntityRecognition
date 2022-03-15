import glob
import json


def progress_print(i, max, append_string, end):
    percent = '({:.2f}%)'.format(i / max * 100)
    print(append_string, i, "of", max, percent, end=end)


file_list = glob.glob('/Users/gabriel-he/Documents/JST data/' + '[!~]*.json')

output_file = open('../data/JST data/症例_filtered.json', 'w')

match_keyword = '症例'

file_count = 0
found = 0
key_count = {}
for file in file_list:
    file_count = file_count + 1
    progress_print(file_count, len(file_list), 'File', '\n')

    line_count = 0
    try:
        for line in open(file):
            line_count = line_count + 1
            print('Line', line_count, end='\r')

            doc = json.loads(line)
            try:
                keywords = doc['タイトル切り出し語(絞り込み用)']
                if not keywords or match_keyword not in keywords:
                    continue

                found = found + 1
                output_file.write(line)
            except KeyError:
                continue
    except Exception as e:
        print(e)
    print("Found: ", found)
