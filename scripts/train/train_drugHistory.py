from BERT.train import train_from_xml_file

xmlFile = '../../data/drugHistoryCheck.xml'
model = 'cl-tohoku/bert-base-japanese-char-v2'
tag_list = ['d']
output_dir = '../../out/out'

train_from_xml_file(xmlFile, model, tag_list, output_dir)
