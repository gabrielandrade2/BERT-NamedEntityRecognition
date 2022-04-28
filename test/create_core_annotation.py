import difflib
import os
from os import path

from tqdm import tqdm

from util import iob_util, text_utils
from util.text_utils import findstem

if __name__ == '__main__':
    directory = '/Users/gabriel-he/Documents/datasets/薬歴'
    annotated_dataset = [f.path for f in os.scandir(directory) if f.is_dir()]
    output_directory = path.join(directory, '薬歴_coreタグ付け')

    # Get list of unique files that exists in all subfolders
    temp_files = [[file for file in os.listdir(folder) if path.isfile(path.join(folder, file))]
                  for folder in annotated_dataset]
    files = set()
    [files.update(x) for x in temp_files]
    files = sorted(files)
    # files = ['1046_Xyzal_Syrup.txt']

    # Iterate over all files, trying to read from all datasets
    for file in tqdm(files):
        texts = []
        for folder in annotated_dataset:
            try:
                with open(path.join(folder, file)) as f:
                    texts.append('\n'.join(f.readlines()).strip())
            except FileNotFoundError:
                pass

        # Skip files with single-person annotation
        if len(texts) < 2:
            continue

        ### Insanity check ###
        # Check if the texts are actually identical without the tags
        temp_texts = [text_utils.remove_tags(text) for text in texts]
        skip = False
        for i in range(1, len(temp_texts)):
            if not (temp_texts[0] == temp_texts[i]):
                skip = True
                break
        if skip:
            print("\n\n\nTexts from file {} are not identical".format(file))

            print('{} \n==================>\n {}'.format(temp_texts[0], temp_texts[1]))
            for i, s in enumerate(difflib.ndiff(temp_texts[0], temp_texts[1])):
                if s[0] == ' ':
                    continue
                elif s[0] == '-':
                    print(u'Delete "{}" from position {}'.format(s[-1], i))
                elif s[0] == '+':
                    print(u'Add "{}" to position {}'.format(s[-1], i))
            print()

            continue

        # After reading all texts, get all the annotations
        tags = []
        for text in texts:
            tags.extend(iob_util.convert_xml_to_taglist(text, ignore_mismatch_tags=False)[1])
        tags = sorted(tags, key=lambda x: x[0])
        # print('\n\n\n', file)
        # print(tags)

        # Compute the span of the core tag
        i = 0
        core_tags = set()
        flag = False
        while i < len(tags):
            tag = tags[i]
            to_process = [tag[3]]
            start = tag[0]
            end = tag[1]

            for j in range(len(tags)):
                if i == j:
                    continue
                if tags[j][0] >= end or tags[j][1] < start:
                    continue

                if start == tags[j][0] and tags[j][1] == end:
                    to_process.append(tags[j][3])
                    i = j

                # Look for tags that overlap
                elif (start <= tags[j][0] < end or start < tags[j][1] <= end):  # FIXME: We have to check the tag type
                    to_process.append(tags[j][3])
                    i = j
                    flag = True

            i += 1
            # If there is only one annotation, do not add the core tag
            if len(to_process) < 2:
                continue

            # Get the longest common substring
            core = findstem(to_process)
            if not core:
                continue
            core_start = tag[3].find(core) + tag[0]
            core_end = core_start + len(core)
            core_tag = (core_start, core_end, 'core', core)
            # print(core_tag)
            core_tags.add(core_tag)

        if flag:
            print('\n\n\n', file)
            print(tags)
            [print(tag) for tag in sorted(core_tags)]

        # Re-tag the texts
        for text in texts:
            tags = iob_util.convert_xml_to_dict(text, ignore_mismatch_tags=False)
            core_tags_dict = []
            for tag in core_tags:
                core_tags_dict.append({
                    "span": (int(tag[0]), int(tag[1])),
                    "word": tag[3],
                    "type": tag[2]
                })
            # tags[1].extend(core_tags_dict)
            iob_util.convert_dict_to_xml(text, tags[1])
