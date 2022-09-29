import glob

import pandas as pd

from util import iob_util
from util.xml_parser import xml_to_article_texts

if __name__ == '__main__':
    for path in glob.glob("/Users/gabriel-he/Documents/oici0928/IM_v6_OICI_finetune_10_epochs_plus3/*.txt"):
        out_path = path.replace('.txt', '.xlsx')

        articles = xml_to_article_texts(path)
        df = pd.DataFrame(columns=['Article #', 'C', 'CN'])
        for i, article in enumerate(articles):
            c = []
            cn = []
            _, entries = iob_util.convert_xml_to_taglist(article, ['C', 'CN'])
            for entry in entries:
                if entry[2] == 'C':
                    c.append(entry[3])
                elif entry[2] == 'CN':
                    cn.append(entry[3])
            df.loc[i] = [i + 1, c, cn]
        df.to_excel(out_path, index=False)

        # df = pd.DataFrame(columns=['id'])
        # i = 0
        # with open(path, 'r') as f:
        #     p = re.compile("<article id=\"(\d+)\"")
        #     for line in f.readlines():
        #         m = p.match(line)
        #         if m:
        #             article_id = m.group(1)
        #             df.loc[i] = [int(article_id)]
        #             i += 1
        # df.to_excel("/Users/gabriel-he/Documents/oici0928/test.xlsx")
