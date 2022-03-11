from xml.dom import minidom

import pandas as pd
from lxml.etree import XMLSyntaxError, XMLParser

from util.iob_util import convert_xml_text_to_iob
from util.text_utils import *


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
    articles = preprocessing(articles)
    f = open("out/iob.iob", 'w')
    article_index = 0
    processed_iob = list()
    for article in articles:
        iob = convert_xml_text_to_iob(article, tag_list)
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


def convert_xml_file_to_iob_list(file, tag_list, should_split_sentences=False, ignore_mismatch_tags=True):
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
            iob = convert_xml_text_to_iob(t, tag_list, ignore_mismatch_tags=ignore_mismatch_tags)
            # Convert tuples into lists
            for item in iob:
                if item[0] == ' ':
                    continue
                sent.append(item[0])
                tag.append(item[1])
            items.append(sent)
            tags.append(tag)
        except XMLSyntaxError:
            print("Skipping text with xml syntax error, id: " + str(i))
        i = i + 1
    return items, tags


def convert_xml_file_to_iob_file(file, tag_list, out_file, ignore_mismatch_tags=True):
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
                iob = convert_xml_text_to_iob(sentence, tag_list, ignore_mismatch_tags=ignore_mismatch_tags)
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
    articles = preprocessing(articles)

    if should_split_sentences:
        texts = split_sentences(articles)
    else:
        texts = articles
    return texts


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

# def entities_from_xml(file_name, attrs = False):#attrs=属性を考慮するか否か，考慮しないならFalse
#     frequent_tags_attrs = select_tags(attrs)
#     dict_tags = dict(zip(frequent_tags_attrs, tags_value))#type_id への変換用
#     with codecs.open(file_name, "r", "utf-8") as file:
#         soup = BeautifulSoup(file, "html.parser")
#
#     for elem_articles in soup.find_all("articles"):#articles内のarticleを一つずつ取り出す
#         entities = []
#         articles = []
#         for elem in elem_articles.find_all('article'):#article内の要素を一つずつ取り出す
#             entities_article = []
#             text_list = []
#             pos1 = 0
#             pos2 = 0
#             for child in elem:#取り出した要素に対して，一つずつ処理する
#                 #（タグのないものについても要素として取得されるので，位置(pos)はずれない）
#                 text = unicodedata.normalize('NFKC', child.string)#正規化
#                 #text = text.replace('。', '.')#句点を'.'に統一, sentenceの分割に使うため．
#                 pos2 += len(text)#終了位置を記憶
#                 if child.name in frequent_tags:#特定のタグについて，固有表現の表現形，位置，タグを取得
#                     attr = ""#属性を入れるため
#                     if 'type' in child.attrs:#typeがある場合には
#                         attr = child.attrs['type']
#                     if 'certainty' in child.attrs:#certaintyがある場合には
#                         attr = child.attrs['certainty']
#                     if 'state' in child.attrs:#stateがある場合には
#                         attr = child.attrs['state']
#                     if not attrs:#attrs=属性を考慮するか否か，考慮しないならFalse
#                         attr = ""
#                     entities_article.append({'name':text, 'span':[pos1, pos2],\
#                         'type_id':dict_tags[str(child.name)+'_'+str(attr)],\
#                         'type':str(child.name)+'_'+str(attr)})
#                 pos1 = pos2#次のentityの開始位置を設定
#                 text_list.append(text)
#             articles.append("".join(text_list))
#             entities.append(entities_article)
#     return articles, entities
