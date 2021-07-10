import enum
import json
from collections import Counter
import itertools

import config
import data
import utils


def extract_vocab(iterable, top_k=None, start=0):
    """ Turns an iterable of list of tokens into a vocabulary.
        These tokens could be single answers or word tokens in questions.
    """
    all_tokens = itertools.chain.from_iterable(iterable)
    counter = Counter(all_tokens)
    if top_k:
        most_common = counter.most_common(top_k)
        most_common = (t for t, c in most_common)
    else:
        most_common = counter.keys()
    # descending in count, then lexicographical order
    tokens = sorted(most_common, key=lambda x: (counter[x], x), reverse=True)
    vocab = {t: i for i, t in enumerate(tokens, start=start)}
    return vocab


def main():
    questions = utils.path_for(train=True, question=True)
    answers = utils.path_for(train=True, answer=True)

    with open(questions, 'r') as fd:
        questions = json.load(fd)
    with open(answers, 'r') as fd:
        answers = json.load(fd)

    questions = data.prepare_questions(questions)
    answers = data.prepare_answers(answers)

    question_vocab = extract_vocab(questions, start=1)
    answer_vocab = extract_vocab(answers, top_k=config.max_answers)

    print("Questions' vocabulary size: ", len(question_vocab))
    print("Top 5 words in questions:")
    for (word, index) in question_vocab.items():
        print("\'{}\'\tid: {}".format(word, index))
        if index > 4: break; 

    print("Answers' vocabulary size: ", len(answer_vocab))

    print("Top 5 answers:")
    for (word, index) in answer_vocab.items():
        print("\'{}\'\tid: {}".format(word, index))
        if index > 3: break;

    vocabs = {
        'question': question_vocab,
        'answer': answer_vocab,
    }
    with open(config.vocabulary_path, 'w') as fd:
        json.dump(vocabs, fd)


if __name__ == '__main__':
    main()
