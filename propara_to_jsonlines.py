import argparse
import json
from transformers import BertTokenizerFast
from bert.tokenization import BasicTokenizer


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('data', type=str)
    parser.add_argument('output_file', type=str)
    parser.add_argument('vocab_file', type=str)

    return parser.parse_args()


def process_data(data_file, output_file, vocab_file):
    """
    Adapted from the `gap_to_jsonlines.py` for propara data prep
    """
    tokenizer = BertTokenizerFast(vocab_file=vocab_file)
    # Need to have this other tokenizer so we can build the sub-token
    # to token map. It seems huggingface tokenizer doesn't have this
    # functionality
    basic_tokenizer = BasicTokenizer(do_lower_case=False)

    # Load data
    with open(data_file, 'r') as fp:
        data = json.load(fp)

    output_jsons = []
    # Format as jsonlines & tokenize
    for para in data:
        output = {}
        paragraph_text = " ".join(para['sentence_texts'])

        # Sentence map
        sentence_map = [0]
        for sent_num, sent in enumerate(para['sentence_texts']):
            tokens = tokenizer.tokenize(sent)
            sentence_map += [sent_num] * len(tokens)
        sentence_map += [sentence_map[-1]]

        # All tokens
        # Note this is the same as what we used to calculate the sentence map
        # even though they are done separately
        tokenized_paragraph = tokenizer(
            paragraph_text, return_offsets_mapping=True)
        paragraph_tokens = tokenizer.batch_decode(
            tokenized_paragraph['input_ids'])
        token_character_offsets = tokenized_paragraph['offset_mapping']

        # Subtoken map
        # 0 element is for CLS
        subtoken_map = [0]
        for tok_id, token in enumerate(basic_tokenizer.tokenize(paragraph_text)):
            subtokens = tokenizer.tokenize(token)
            subtoken_map += [tok_id] * len(subtokens)
        # Add on last subtoken for SEP
        subtoken_map += [subtoken_map[-1]]

        output['speakers'] = ['[SPL]'] + ['-'] * \
            (len(paragraph_tokens) - 2) + ['[SPL]']
        output['sentences'] = paragraph_tokens
        output['sentence_map'] = sentence_map
        output['clusters'] = [[]]
        output['subtoken_map'] = subtoken_map
        output['token_char_spans'] = token_character_offsets
        output['original_text'] = paragraph_text

        print(list(zip(paragraph_tokens, subtoken_map, sentence_map)))
        # Test, if we know we have a mention on tokens 2-8
        # how do we translate that to a span in the original sentence?
        output_jsons.append(output)

    # output to output_file
    with open(output_file, 'w') as fp:
        for out in output_jsons:
            fp.write(json.dumps(out) + '\n')


if __name__ == "__main__":
    ARGS = parse_args()

    process_data(ARGS.data, ARGS.output_file, ARGS.vocab_file)
