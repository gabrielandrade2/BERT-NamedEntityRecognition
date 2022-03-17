import sqlite3
from sqlite3 import Error

import pandas as pd
from thefuzz import fuzz

from util.text_utils import EntityNormalizerInterface


class MedDRADatabase:

    def __init__(self, db_file):
        self.db_file = db_file
        self.conn = None

    def open_connection(self):
        """ Opens the connection to the SQLite database. """
        conn = None
        try:
            conn = sqlite3.connect(self.db_file)
        except Error as e:
            print(e)
            self.close_connection()

        self.conn = conn

    def close_connection(self):
        """ Closes a current open connection to the SQLite database. """
        if self.conn:
            self.conn.close()
        self.conn = None

    def get_pt_j_from_llt_code(self, llt_code):
        """ Get the Japanese preferred term (PT) for a given Low-level term (LLT) code.

        :param llt_code: The llt_code of the term.
        :return: The string containing the Japanese PT or None, if a match is not found.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            'SELECT pt_kanji FROM llt INNER JOIN pt_j ON llt.pt_code = pt_j.pt_code WHERE llt_code = {}'.format(
                llt_code))
        res = cursor.fetchall()

        if not res:
            return None
        else:
            return res[0][0]


class MedDRAPatientFriendlyList:

    def __init__(self, xls_file_path):
        self.df = pd.read_excel(xls_file_path)

    def get_dataframe(self):
        return self.df

    def get_patient_friendly_tuples(self):
        return list(self.df[['LLT kanji', 'LLT code', 'Level']].itertuples(index=False, name=None))


class MedDRAPatientFriendlyPTEntityNormalizer(EntityNormalizerInterface):

    def __init__(self, database, patient_friendly_list):
        self.database = database
        self.patient_friendly_list = patient_friendly_list
        self.pf_list = patient_friendly_list.get_patient_friendly_tuples()

    def normalize(self, term, matching_method=fuzz.token_set_ratio, threshold=0):
        preferred_candidate = max([(matching_method(term, pf[0]), pf) for pf in self.pf_list])
        patient_friendly_term = preferred_candidate[1]
        score = preferred_candidate[0]

        if score > threshold:
            if patient_friendly_term[2] == 'pt':
                normalized_term = patient_friendly_term[0]
            else:
                normalized_term = self.database.get_pt_j_from_llt_code(patient_friendly_term[1])
        else:
            normalized_term = term

        return normalized_term, score
