import pandas as pd

file = '../data/DATA_IM_v6.txt'
data = pd.read_csv(file, sep="	")
texts_tagged = data['text_tagged'].tolist()
texts_raw = data['text_raw'].tolist()
