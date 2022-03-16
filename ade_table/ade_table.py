import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def from_lists(cls, drugs, entities):
    assert len(drugs) == len(entities)
    output_dict = {}

    for drug, entity_list in zip(drugs, entities):
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

    def table_post_process(self, inplace=False):
        if inplace:
            table = self.table
        else:
            table = pd.DataFrame(self.table)

        # Order drugs by number of ADE events
        table['sum_col'] = table.sum(axis=1)
        table.sort_values('sum_col', axis=0, ascending=False, inplace=True)
        table.drop(columns=["sum_col"], inplace=True)
        # table = table[:50]

        # Order ADE by numer of events
        # table = table[table.sum(0).sort_values(ascending=False)[:50].index]
        table = table[table.sum(0).sort_values(ascending=False).index]

        return table

    def generate_heatmap(self):
        mpl.rc('font', family="Hiragino Sans")
        sns.set(font='Hiragino Sans')
        sns.color_palette("YlOrBr", as_cmap=True)
        plt.figure(figsize=(20, 20))
        heatmap = sns.heatmap(self.table)
        heatmap.figure.tight_layout()
        fig = heatmap.get_figure()
        fig.savefig("out.png")
        plt.show()
