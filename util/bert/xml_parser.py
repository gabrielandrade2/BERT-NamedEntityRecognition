import re
import pandas as pd

from xml.dom import minidom
from lxml.etree import XMLSyntaxError, XMLParser
from util.bert.iob_util import convert_xml_to_iob


def xml_to_articles(file):
    """Extract all instances of <article> into a list, from a given xml file.

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


def convert_xml_to_iob_list(file, tag_list, should_split_sentences=False, ignore_mismatch_tags=True):
    """Converts a corpus xml file to a tuple of strings and IOB tags.
    The strings can be split by article or sentences.

    :param file: The XML file to be parsed.
    :param tag_list: The list of tags to be extracted from the file.
    :param should_split_sentences: Should articles be split into sentences?
    :param ignore_mismatch_tags: Should skip texts that contain missing/mismatch xml tags
    :return: The list of strings and the list of IOB tags
    """

    # Preprocess
    texts = __prepare_texts(file, should_split_sentences)

    # Convert
    items = list()
    tags = list()
    i = 0
    for t in texts:
        sent = list()
        tag = list()
        try:
            iob = convert_xml_to_iob(t, tag_list, ignore_mismatch_tags=ignore_mismatch_tags)
            # Convert tuples into lists
            for item in iob:
                if item[0] == ' ':
                    continue
                sent.append(item[0])
                tag.append(item[1])
            items.append(sent)
            tags.append(tag)
            i = i + 1
        except XMLSyntaxError:
            print("Skipping text with xml syntax error, id: " + str(i))
    return items, tags


def convert_xml_to_iob_file(file, tag_list, out_file, ignore_mismatch_tags=True):
    """Converts a corpus xml file into IOB2 format and save it to a file in CONLL 2003 format.

    :param file: The XML file to be parsed.
    :param tag_list: The list of tags to be extracted from the file.
    :param out_file: The output path for the .iob file
    :param ignore_mismatch_tags: Should skip texts that contain missing/mismatch xml tags
    """

    # Preprocess
    texts = __prepare_texts(file, False)
    texts = split_sentences(texts, False)

    if not out_file.endswith('.iob'):
        out_file.append('iob')

    try:
        f = open(out_file, 'w')
    except OSError:
        print("Failed to open file for writing: " + out_file)
        return
    for text in texts:
        for sentence in text:
            try:
                iob = convert_xml_to_iob(sentence, tag_list, ignore_mismatch_tags=ignore_mismatch_tags)
                f.write('\n'.join('{}\t{}'.format(x[0], x[1]) for x in iob))
                f.write('\n\n')
            except XMLSyntaxError:
                print("Skipping sentence with xml syntax error")

def __prepare_texts(file, should_split_sentences):
    """ Loads a file and applies all the preprocessing steps before format conversion.

    :param file: The xml file to be loaded.
    :param should_split_sentences: Should the sentences from an article be split.
    :return: The list of string with the desired format.
    """
    articles = xml_to_articles(file)
    articles = __preprocessing(articles)

    if should_split_sentences:
        texts = split_sentences(articles)
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


def split_sentences(texts, return_flat_list=True):
    """Given a list of strings, split them into sentences and join everything together into a flat list containing all
     the sentences.

    :param texts: List of strings to be processed.
    :param return_flat_list: If True return a flat list with all the sentences, otherwise a list of lists.
    :return: The list of split sentences.
    """
    processed_texts = list()
    for text in texts:
        processed_text = re.split(
                    "(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.?!])\s\n*|(?<=[^A-zＡ-ｚ0-9０-９ ].)(?<=[。．.?？!！])(?![\.」])\n*", text)
        #processed_text = re.split("(? <=[。?？!！])")  # In case only a simple regex is necessary
        processed_text = [x.strip() for x in processed_text]
        processed_text = [x for x in processed_text if x != '']
        if return_flat_list:
            processed_texts.extend(processed_text)
        else:
            processed_texts.append(processed_text)
    return processed_texts

def drop_texts_with_mismatched_tags(texts):
    no_mismatch = list()
    for text in texts:
        try:
            tagged_text = '<sent>' + text + '</sent>'
            parser = XMLParser()
            parser.feed(tagged_text)
            no_mismatch.append(text)
        except XMLSyntaxError:
            continue
    return no_mismatch
