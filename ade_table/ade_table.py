import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from util.text_utils import EntityNormalizer


def from_lists(drugs: list, entities: list, normalization_model: EntityNormalizer = None):
    assert len(drugs) == len(entities)
    output_dict = {}

    if normalization_model:
        entities = [normalization_model.normalize_list(entity_list)[0] for entity_list in entities]

    for drug_list, entity_list in zip(drugs, entities):
        for drug in drug_list:
            # Add new drug to table
            if drug in output_dict:
                drug_dict = output_dict[drug]
            else:
                drug_dict = {}

            # Check if this entity exists for the current drug row
            for named_entity in entity_list:
                if named_entity in drug_dict:
                    count = drug_dict[named_entity] + 1
                else:
                    count = 1
                drug_dict[named_entity] = count
            output_dict[drug] = drug_dict

    return ADETable(output_dict)

class ADETable:

    def __init__(self, data):
        self.table = pd.DataFrame.from_dict(data, orient='index').fillna(0)
        self.__post_process()

    def __post_process(self):
        # Order drugs by number of ADE events
        self.table['sum_col'] = self.table.sum(axis=1)
        self.table.sort_values('sum_col', axis=0, ascending=False, inplace=True)
        self.table.drop(columns=["sum_col"], inplace=True)
        # table = table[:50]

        # Order ADE by numer of events
        # table = table[table.sum(0).sort_values(ascending=False)[:50].index]
        self.table = self.table[self.table.sum(0).sort_values(ascending=False).index]

    def filter(self, num_drugs=None, num_symptoms=None):
        table = pd.DataFrame(self.table)
        if num_drugs:
            table = table[:num_drugs]
        if num_symptoms:
            table = table[table.sum(0).sort_values(ascending=False)[:num_symptoms].index]
        return table

    def to_dataframe(self):
        return self.table

    def generate_heatmap(self, output_path):
        mpl.rc('font', family="Hiragino Sans")
        sns.set(font='Hiragino Sans')
        sns.color_palette("YlOrBr", as_cmap=True)
        plt.figure(figsize=(20, 20))
        heatmap = sns.heatmap(self.table)
        heatmap.figure.tight_layout()
        if output_path:
            fig = heatmap.get_figure()
            fig.savefig(output_path)
        plt.show()