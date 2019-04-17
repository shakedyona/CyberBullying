from gensim.corpora import WikiCorpus
from Embedding.word2vec import get_model, train
import csv


def create_wiki_corpus(inp="hewiki-latest-pages-articles.xml.bz2", outp="wiki3.he.text"):
    print("Starting to create wiki corpus")
    output = open(outp, 'w')
    space = " "
    wiki = WikiCorpus(inp, lemmatize=False, dictionary={})
    for i, text in enumerate(wiki.get_texts()):
        article = space.join([t for t in text])

        output.write("{}\n".format(article))
        if i % 1000 == 0:
            print("Saved " + str(i) + " articles")

    output.close()


def get_word_vector(model, word):
    return model.wv[word]


def create_our_corpus(inp='corpus.csv', outp='corpus.txt'):
    with open(inp) as csvfile, open(outp, 'w') as txtFile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            txtFile.write(' '.join(row))
    print('done..')


m = get_model("our.corpus.word2vec.model")

print(get_word_vector(m, "ימח"))