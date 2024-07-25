from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from clams_utils.aapb.newshour_transcript_cleanup import file_cleaner, clean
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import os
import argparse
import nltk
nltk.download('wordnet')
stop_words = (stopwords.words('english')
              + ['com', 'edu', 'subject', 'lines', 'organization', 'would', 'article', 'could'])


def tokenizer(transcripts_list: list):
    """Given a list of list(s) of transcripts, tokenize the transcripts."""
    for idx in range(len(transcripts_list)):
        # convert into lowercase first, then tokenize
        transcripts_list[idx] = RegexpTokenizer(r'\w+').tokenize(transcripts_list[idx].lower())

    # Remove numbers, but not words that contain numbers.
    docs = [[token for token in transcript if not token.isnumeric()] for transcript in transcripts_list]
    # Remove words that are only one character.
    docs = [[token for token in doc if len(token) > 1] for doc in docs]

    return docs


def lemmatizer(token_lists: list):
    """
    Given a list of list(s) of tokens, lemmatize the tokens.
    """
    # Lemmatize the documents.
    text = [[WordNetLemmatizer().lemmatize(token) for token in token_list if token not in stop_words] for token_list in token_lists]

    return text


def read_newshour_transcript(newshour_transcripts_directory):
    """
    Given a directory of NewsHour transcripts,
    return a list that contains all the cleaned transcripts for later model training.
    """
    all_text_list = []
    for txt_file in os.listdir(newshour_transcripts_directory):
        txt_file_path = os.path.join(newshour_transcripts_directory, txt_file)
        if txt_file_path.endswith('.txt'):
            cleaned_lines = file_cleaner(txt_file_path)
            if cleaned_lines is not None:
                all_text_list.append(cleaned_lines)

    return all_text_list


def nh_transcripts_preprocessor(nh_transcript_directory):
    """
    Preprocess all NewsHour transcripts in a given directory into dictionary and corpus for LDA model training.
    """
    transcripts_list = read_newshour_transcript(nh_transcript_directory)

    # tokenize the text
    texts = tokenizer(transcripts_list)

    # lemmatize the text
    texts = lemmatizer(texts)

    dictionary = Dictionary(texts)

    corpus = [dictionary.doc2bow(text) for text in texts]

    return dictionary, corpus


def file_processor(text: str):
    """
    Preprocess a text string into dictionary and corpus for keyword extraction.
    """
    text = clean(text)
    text = tokenizer([text])
    text = lemmatizer(text)
    dictionary = Dictionary(text)
    corpus = [dictionary.doc2bow(text[0])]
    return dictionary, corpus


def training(dictionary, corpus, num_topics=1, model_name="model"):
    """Train an LDA model."""
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
    temp_file = "./" + model_name
    lda.save(temp_file)

    return temp_file


def extract_keyword_coherence_pairs(pretrained_model_dir, text, topn=10):
    """
    Given the directory of the pretrained LDA model and a text string that is not part of the corpus,
    return the topn keywords of the text string,
    """
    id2word_dict = Dictionary.load('model.id2word')
    lda = LdaModel.load(pretrained_model_dir)
    new_dict, new_corpus = file_processor(text)
    topics = sorted(lda.get_document_topics(new_corpus[0]), key=lambda x: x[1], reverse=True)
    top_topic_terms = lda.get_topic_terms(topics[0][0], topn=topn)
    keyword_coherence_pairs = []
    for term in top_topic_terms:
        keyword = id2word_dict[term[0]]
        # cast the coherence score (numpy.float32) into float or TextDocument won't accept such value
        coherence = float(term[1])
        keyword_coherence_pairs.append((keyword, coherence))
    return keyword_coherence_pairs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataPath", action='store',
                        help='path to the directory of all transcripts used for training the LDA model.')
    # TODO: comment this argument back after the modelName parameter in metadata.py is solved
    # parser.add_argument("--modelName", action='store', default='model',
    #                     help='file name of the pretrained LDA model.')
    parser.add_argument("--numTopics", action='store', type=int, default=1,
                        help='number of topics to extract when training the LDA model.')
    parsed_args = parser.parse_args()
    dictionary, corpus = nh_transcripts_preprocessor(parsed_args.dataPath)
    # TODO: add the additional parameter, parsed_args.modelName, once the previous TODO is solved
    training(dictionary, corpus, num_topics=parsed_args.numTopics)
