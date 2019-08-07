from gensim.corpora import WikiCorpus
import csv


def create_wiki_corpus(inp="hewiki-latest-pages-articles.xml.bz2", outp="wiki3.he.text"):
    """
    create file of text line from xml files
    :param inp:
    :param outp:
    :return:
    """
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


def create_our_corpus(inp='corpus.csv', outp='corpus.txt'):
    """
    create text file from csv file
    :param inp:
    :param outp:
    :return:
    """
    with open(inp) as csvfile, open(outp, 'w') as txtFile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            txtFile.write(' '.join(row))
    print('done..')


def create_keepers_corpus(inp_neg, inp_pos, out='keepers_corpus.txt'):
    """
    create one text file from all the post in negative text file and positive text file from 'Keepers'
    :param inp_neg:
    :param inp_pos:
    :param out:
    :return:
    """
    with open(inp_neg) as negfile, open(inp_pos) as posfile, open(out, 'w') as outfile:
        neg_lines = negfile.readlines()
        pos_lines = posfile.readlines()
        for line in neg_lines:
            outfile.write(line)
        for line in pos_lines:
            outfile.write(line)
    print('done..')
