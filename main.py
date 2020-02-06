import math
import pickle
import re
from collections import defaultdict
from pathlib import Path
import requests
import dill

data_dir = Path('./data/text')
data_pairs_dir = Path('./data/pairs')
adj_for_category = Path('categories/adj_for_category.txt')
CREATE_DICT = False
CREATE_CATEGORY_VECTORS = False


def main():
    if CREATE_DICT:
        for file in data_dir.iterdir():
            if not (data_pairs_dir / file.name).exists():
                find_pairs_adj_noun(file)

        nouns_adjs = defaultdict(lambda: defaultdict(float))

        for file in data_pairs_dir.iterdir():
            with file.open('r') as file_with_pairs:
                for line in file_with_pairs.readlines():
                    noun_adj = line.replace('\n', '').split('#')
                    nouns_adjs[noun_adj[0]][noun_adj[1]] += 1

        for key, adj_dict in nouns_adjs.items():
            norm_value = norm(adj_dict)
            for adj_key, adj_value in nouns_adjs[key].items():
                nouns_adjs[key][adj_key] = adj_value / norm_value

        with open('obj/dict_noun_adj.pkl', 'wb+') as f:
            pickle.dump(dict(nouns_adjs), f, pickle.HIGHEST_PROTOCOL)

    else:
        with open('obj/dict_noun_adj.pkl', 'rb') as f:
            nouns_adjs = pickle.load(f)

    if CREATE_CATEGORY_VECTORS:
        categories_vector = create_vectors_for_all_category(nouns_adjs)
    else:
        with open('obj/categories_vector.pkl', 'rb') as f:
            categories_vector = pickle.load(f)

    # enter word
    while True:
        print('Wprowadź słowo:')
        word = input()

        noun_vector = nouns_adjs.get(word, '')
        if noun_vector == '':
            print('Nie ma takiego słowa w bazie')
        else:
            norm_noun_vector = norm(noun_vector)
            max_adj_for_noun = ''
            max_number_for_adj = 0
            for adj, value in noun_vector.items():
                if value > max_number_for_adj:
                    max_adj_for_noun = adj
                    max_number_for_adj = value

            result = {}
            for key, category_dict in categories_vector.items():
                norm_category = norm(category_dict)

                tmp_vector = {k: noun_vector.get(k, 0) * category_dict.get(k, 0)
                              for k in set(noun_vector) | set(category_dict)}
                if norm_category != 0 and norm_noun_vector != 0:
                    result[key] = sum(tmp_vector.values()) / (norm_noun_vector * norm_category)
                else:
                    result[key] = 0

            print(f'Najczęściej występujący przymotnik: {max_adj_for_noun}')
            for category, value in {k: v for k, v in
                                    sorted(result.items(), key=lambda item: item[1], reverse=True)}.items():
                print(f'{category} {str(value)}', end=' | ')
            print('\n')


def norm(dictionary):
    sum_vector = 0.0
    for key, value in dictionary.items():
        sum_vector += value * value
    sum_vector = math.sqrt(sum_vector)
    return sum_vector


def find_pairs_adj_noun(file):
    look_ahead_word = 3
    adjs = {}
    nouns = {}
    with open('categories/class_adj.txt', 'r') as adj_file:
        for line in adj_file.readlines():
            adjs[line.split(';')[0]] = 0

    with open('categories/class_noun.txt', 'r') as noun_file:
        for line in noun_file.readlines():
            nouns[line.split(';')[0]] = 0

    with open(file, 'r') as text_file:
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text_file.read().replace('\n', ''))
        pairs_noun_adj = []
        for sentence in sentences:
            response = requests.post('http://localhost:9003', data=sentence.encode('utf-8'))
            elem = response.text.split('\n')
            if len(elem) < 5:
                continue
            tokens = elem[1:-4:2]
            words_with_tags = []

            for token in tokens:
                word_tag = token.split(':')[0]
                clean_word_tag = word_tag.split('\t')[1:]
                if len(clean_word_tag) == 2:
                    words_with_tags.append((clean_word_tag[0], clean_word_tag[1]))

            for i, (word, tag) in enumerate(words_with_tags):
                if tag == 'adj':
                    for ahead in range(1, look_ahead_word + 1):
                        if i + ahead >= len(words_with_tags):
                            break
                        else:
                            if words_with_tags[i + ahead][1] == 'subst':
                                noun = words_with_tags[i + ahead][0]
                                if noun in nouns and word in adjs:
                                    pairs_noun_adj.append(f'{noun}#{word}')
                                    break
        with open(data_pairs_dir / file.name, 'w+') as file_to_write:
            for pair in pairs_noun_adj:
                file_to_write.write(pair + '\n')


def create_vectors_for_all_category(nouns_adjs):
    with open('categories/semantic_groups.txt') as categories_example_file:
        categories_vector = {}
        for line in categories_example_file:
            words_list = line.replace('\n', '').split('#')
            categories_vector[words_list[0]] = create_vector_for_category(words_list[0], words_list[1:], nouns_adjs)

    with open('obj/categories_vector.pkl', 'wb+') as f:
        pickle.dump(categories_vector, f, pickle.HIGHEST_PROTOCOL)

    return categories_vector


def create_vector_for_category(category_name, examples, nouns_adjs):
    vector = {}
    for word in examples:
        vector = {
            k: vector.get(k, 0) + nouns_adjs[word].get(k, 0)
            for k in set(vector) | set(nouns_adjs[word])
        }
    with open('categories/adj_for_category.txt', 'a+') as f:
        sort_vector = [k for k, v in sorted(vector.items(), key=lambda item: item[1], reverse=True)]
        f.write(category_name)
        for elem in sort_vector:
            f.write('#' + elem)
        f.write('\n')

    norm_vector = norm(vector)
    return {key: (value / norm_vector) for key, value in vector.items()}


if __name__ == '__main__':
    main()

# run docker for tagging and lemmatization
# systemctl start docker
# sudo docker run -p 9003:9003 -it djstrong/krnnt:1.0.0
