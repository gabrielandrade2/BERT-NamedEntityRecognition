import re
import pandas as pd
from xml.dom import minidom
from mojimoji import han_to_zen
from BERT.util.iob_util import convert_xml_to_iob


def xml_to_articles(file):
    """Extract all instances of <article> into a list from a given xml file.

    :param file: The corpus xml file.
    :return: List of strings, containing all the articles as found in the file.
    """

    xml_document = minidom.parse(file)
    docs = xml_document.getElementsByTagName('article')
    print('Converting ' + str(len(docs)) + ' articles')

    articles = list()
    for doc in docs:
        text = ''
        for i in doc.childNodes:
            text += i.toxml()
        articles.append(text)
    return articles


def convert_xml_to_dataframe(file, tag_list, print_info=True):
    """ Converts a corpus xml file to a dataframe.

    :param file:
    :param tag_list:
    :param print_info:
    :return: The converted dataframe.
    """

    # Preprocess
    articles = xml_to_articles(file)
    articles = __preprocessing(articles)
    f = open("out/iob.iob", 'w')
    article_index = 0
    processed_iob = list()
    for article in articles:
        iob = convert_xml_to_iob(article, tag_list)
        f.write('\n'.join('{}	{}'.format(x[0], x[1]) for x in iob))
        for item in iob:
            processed_iob.append((article_index,) + item)
        f.write('\n')
        article_index = article_index + 1
    df = pd.DataFrame(processed_iob, columns=['Article #', 'Word', 'Tag'])

    # Print some info
    if print_info:
        print(df.head())
        print("Number of tags: {}".format(len(df.Tag.unique())))
        print(df.Tag.value_counts())

    return df


def convert_xml_to_iob_list(file, tag_list, split_sentences=False, romaji_to_zen=False):
    """Converts a corpus xml file to a tuple of strings and IOB tags.
    The strings can be split by article or sentences.

    :param file: The XML file to be parsed.
    :param tag_list: The list of tags to be extracted from the file.
    :param split_sentences: Should articles be split into sentences?
    :param romaji_to_zen: Convert all romaji to Zenkaku?
    :return: The list of strings and the list of IOB tags
    """

    # Preprocess
    texts = __prepare_texts(file, split_sentences)

    # Convert
    items = list()
    tags = list()
    for t in texts:
        sent = list()
        tag = list()
        iob = convert_xml_to_iob(t, tag_list)
        # Convert tuples into lists
        for item in iob:
            if romaji_to_zen:
                sent.append(han_to_zen(item[0], kana=False))
            else:
                sent.append(item[0])
            tag.append(item[1])
        items.append(sent)
        tags.append(tag)
    return items, tags

def convert_xml_to_iob_file(file, tag_list, out_file, split_sentences=False, romaji_to_zen=False):
    """Converts a corpus xml file into IOB format and save it to a file in CONLL 2003 format.

    :param file: The XML file to be parsed.
    :param tag_list: The list of tags to be extracted from the file.
    :param out_file: The output path for the .iob file
    :param split_sentences: Should articles be split into sentences?
    :param romaji_to_zen: Convert all romaji to Zenkaku?
    :return:
    """

    # Preprocess
    texts = __prepare_texts(file, split_sentences)

    if not out_file.endswith('.iob'):
        out_file.append('iob')

    try:
        f = open(out_file, 'w')
    except OSError:
        print("Failed to open file for writing: " + out_file)
        return
    for text in texts:
        iob = convert_xml_to_iob(text, tag_list)
        f.write('\n'.join('{}\t{}'.format(x[0], x[1]) for x in iob))
        f.write('\n')


def __prepare_texts(file, split_sentences):
    """ Loads a file and applies all the preprocessing steps before format conversion.

    :param file: The xml file to be loaded.
    :param split_sentences: Should the sentences from an article be split.
    :return: The list of string with the desired format.
    """
    articles = xml_to_articles(file)
    articles = __preprocessing(articles)

    if split_sentences:
        texts = __split_sentences(articles)
    else:
        texts = articles
    return texts


def __preprocessing(texts, remove_core_tag=True):
    """Preprocessing steps for strings.
    Strip strings, remove <core> tags (for now).

    :param texts: List of strings to be processed.
    :param remove_core_tag: Should remove core tag from the texts?
    :return: The list of processed texts.
    """

    processed_articles = list()
    for text in texts:
        if remove_core_tag:
            # Remove all <core> and </core> for now
            text = text.replace('<core>', '')
            text = text.replace('</core>', '')

        # Remove all \n from the beginning and end of the sentences
        text = text.strip()
        processed_articles.append(text)
    return processed_articles


def __split_sentences(texts):
    """Given a list of strings, split them into sentences and join everything together into a flat list containing all
     the sentences.

    :param texts: List of strings to be processed.
    :return: The list of split sentences.
    """

    flat_processed_texts = list()
    for text in texts:
        processed_text = re.split(
                "(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.?!])\s(?!$)|(?<=[一-龯ぁ-ゔゞァ-・ヽヾ゛゜ー].)(?<=[。.?？!！])(?!\.)(?!$)", text)
        processed_text = [x.strip() for x in processed_text]
        flat_processed_texts.extend(processed_text)
    return flat_processed_texts
