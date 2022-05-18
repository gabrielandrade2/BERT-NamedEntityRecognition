import csv
import glob
import json


def progress_print(i, max, append_string, end):
    percent = '({:.2f}%)'.format(i / max * 100)
    print(append_string, i, "of", max, percent, end=end)


file_list = glob.glob('/Users/gabriel-he/Documents/JST data/2022-05/' + '[!~]*.json')

file_count = 0
key_count = {}
keywords_count = {}
for file in file_list:
    file_count = file_count + 1
    progress_print(file_count, len(file_list), 'File', '\n')

    line_count = 0
    try:
        for line in open(file):
            line_count = line_count + 1
            print('Line', line_count, end='\r')

            items = json.loads(line).items()
            for key, value in items:
                if key not in key_count:
                    key_count[key] = 0
                if value:
                    key_count[key] = key_count[key] + 1

            keywords = dict(items)['タイトル切り出し語(絞り込み用)']
            if keywords:
                for key in keywords:
                    if key not in keywords_count:
                        keywords_count[key] = 0
                    if value:
                        keywords_count[key] = keywords_count[key] + 1

    except Exception as e:
        print(e)

with open('/Users/gabriel-he/Documents/JST data/2022-05/key_count.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['key', 'count'])
    for key, value in key_count.items():
        writer.writerow([key, value])

import operator

keywords_count = sorted(keywords_count.items(), key=operator.itemgetter(1), reverse=True)
with open('/Users/gabriel-he/Documents/JST data/2022-05/keyword_count.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['keyword', 'count'])
    for key, value in keywords_count:
        writer.writerow([key, value])
