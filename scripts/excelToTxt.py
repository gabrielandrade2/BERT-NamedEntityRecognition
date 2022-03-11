import glob
import os
import re

import mojimoji
import pandas as pd
import xlrd


def normalizeText(text):
    text = mojimoji.zen_to_han(text, kana=False)
    return re.search(r'[一-龯ぁ-ゔゞァ-・ヽヾ゛゜ーA-z -]*', text).group()


def excelToTxt(directory, drugDict):
    # Load the model and file list
    file_list = glob.glob(directory + '/*.xlsx')
    folder = os.path.join(directory, "txt-jp-drug")
    os.makedirs(folder, exist_ok=True)
    count = 1

    for file in file_list:
        print(file)
        wb = xlrd.open_workbook(file)
        sh = wb.sheet_by_name("Sheet1")

        # Start in line 3 to ignore column titles and the 例 line
        for rownum in range(2, sh.nrows):
            row = sh.row_values(rownum)

            # get the text and see if is not empty
            text = str(row[2])
            if not text:
                continue

            # Add \n after "。" which do not already have it
            text = re.sub('。(?=[^\n])', "。\n", text)

            # In case it is empty, iterate back until we find the last drug listed
            i = 0
            drug = ""
            while not drug:
                # get drug name
                drug = str(sh.row_values(rownum - i)[1])
                drug = re.search(r'[一-龯ぁ-ゔゞァ-・ヽヾ゛゜ーA-zＡ-ｚ0-9０-９]*', drug).group()
                i = i + 1

            # Replace the drug name with the dictionary one, if it exists
            foundList = drugDict[drugDict['出現形'].str.contains(drug)].index.tolist()
            if foundList:
                for item in foundList:
                    newDrug = drugDict['一般名'].loc[item]
                    if newDrug == newDrug and newDrug != '[AMB]':
                        print("Replacing: ", drug, " with: ", newDrug)
                        drug = newDrug
                        break

            # Normalize drug for filename standards and remove dosage
            drug = normalizeText(drug)
            drug = drug.title()
            drug = drug.strip().replace(' ', '_')
            drug = re.sub(r'(?u)[^-\w.]', '', drug)

            # Normalize metadata
            id = f'{count:04d}'
            ade = normalizeText(row[4])
            local = normalizeText(row[5])

            # Create file name
            filename = id + "_" + drug + ".txt"

            # Convert text to NFKC standard
            # text = mojimoji.han_to_zen(text, kana=False)
            # text_nfkc = text

            # When everything is okay, write the text to its own txt file
            with open(os.path.join(folder, filename), 'w', encoding='utf-8') as f:
                f.write("%" + id + "," + drug + "," + ade + "," + local + "\n")
                f.write(text)
                f.close()
            count = count + 1


def loadDrugDict(dictFile):
    df = pd.read_csv(dictFile, sep=',', header=0)
    df = df[['出現形', '一般名', '出現形英語（DeepL翻訳）']]

    # Remove lines with AMB or NON
    # df.dropna(axis=0, how='any', inplace=True)
    # df.drop(df[df['一般名'] == '[AMB]'].index, inplace=True)
    # df.drop(df[df['一般名'] == '[NON]'].index, inplace=True)

    return df


def main():
    drugDict = loadDrugDict("/Users/gabriel-he/Documents/NAIST-PhD/HYAKUYAKU_FULL_v20210706.csv")
    print(drugDict)
    excelToTxt("../data/Croudworks薬歴", drugDict)


main()
