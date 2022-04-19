from knowledge_bases.meddra import MedDRADatabase

if __name__ == '__main__':
    database = MedDRADatabase('/Users/gabriel-he/Documents/git/meddra-sqlite/db/meddra.sqlite3')
    database.open_connection()

    df = database.get_all_en_jp_llt()
    df['is_pt'] = df['llt_code'] == df['pt_code']
    df.to_excel('MedDRA_llt_en_jp.xlsx')
