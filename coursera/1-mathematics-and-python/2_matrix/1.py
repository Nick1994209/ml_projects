import functools
import re
import numpy as np
from scipy.spatial.distance import cosine


def tokenazier(sentence):
    return [word for word in re.split('[^a-z]', sentence.lower()) if word]


def main():
    with open('sentences.txt', encoding='utf-8') as f:
        sentences_words = list(map(tokenazier, f.readlines()))

    words = set(word for sentence in sentences_words for word in sentence)
    words_indexes = {word: index for index, word in enumerate(words)}

    sentences_matrix = np.zeros([len(sentences_words), len(words)])
    for sentence_index, sentence in enumerate(sentences_words):
        for word in sentence:
            sentences_matrix[sentence_index, words_indexes[word]] += 1

    print(sentences_matrix.shape)
    check_sentence(words_indexes, sentence_indexes=sentences_matrix[0])

    closest_sentences = get_closet_sentences(sentences_matrix[0], sentences_matrix)
    print_closest_sentences(sentences_words, closest_sentences)

    answers = [str(s['index']) for s in closest_sentences[1:3]]
    print('answers:', answers)
    with open('submission-1.txt', 'w', encoding='utf-8') as f:
        f.write(' '.join(answers))


def check_sentence(words_indexes, sentence_indexes):
    indexes_words = {index: word for word, index in words_indexes.items()}
    for word_index, count in enumerate(sentence_indexes):
        if not count:
            continue
        print('%s: %d;' % (indexes_words[word_index], count), end=' ')
    print('\n')


def get_closet_sentences(to_sentence, sentences_matrix):
    close_sentences = []
    for index, sentence in enumerate(sentences_matrix):
        close_sentences.append({'index': index, 'distance': cosine(sentence, to_sentence)})

    return sorted(close_sentences, key=lambda x: x['distance'])


def print_closest_sentences(sentences_words, closest_sentences):
    print('\n\n', sentences_words[0], '\n')

    for close_sentence in closest_sentences:
        print(close_sentence, sentences_words[close_sentence['index']])


if __name__ == '__main__':
    main()
