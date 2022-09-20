import argparse

from util import iob_util, xml_parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare XML tagged files')
    parser.add_argument('--gold', type=str, help='Gold file path', required=True)
    parser.add_argument('--predicted', type=str, help='Predicted file path', required=True)
    parser.add_argument('--tags', type=str, nargs='+', help='XML tags', required=True)
    parser.add_argument('--attr', type=str, nargs='+', help='XML tag attributes', required=False, default=None)
    args = parser.parse_args()

    _, gold = xml_parser.convert_xml_file_to_iob_list(args.gold, args.tags, args.attr, ignore_mismatch_tags=False)
    _, predicted = xml_parser.convert_xml_file_to_iob_list(args.predicted, args.tags, args.attr,
                                                           ignore_mismatch_tags=False)

    iob_util.evaluate_performance(gold, predicted)
