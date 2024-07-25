"""
DELETE THIS MODULE STRING AND REPLACE IT WITH A DESCRIPTION OF YOUR APP.

app.py Template

The app.py script does several things:
- import the necessary code
- create a subclass of ClamsApp that defines the metadata and provides a method to run the wrapped NLP tool
- provide a way to run the code as a RESTful Flask service


"""

import argparse
import logging

# Imports needed for Clams and MMIF.
# Non-NLP Clams applications will require AnnotationTypes

from clams import ClamsApp, Restifier
from mmif import Mmif, AnnotationTypes, DocumentTypes
from lda import stop_words, extract_keyword_coherence_pairs


class GensimTmKeywordextractor(ClamsApp):

    def __init__(self):
        super().__init__()

    def _appmetadata(self):
        # see https://sdk.clams.ai/autodoc/clams.app.html#clams.app.ClamsApp._load_appmetadata
        # Also check out ``metadata.py`` in this directory.
        # When using the ``metadata.py`` leave this do-nothing "pass" method here.
        pass

    def _annotate(self, mmif: Mmif, **parameters) -> Mmif:
        self.mmif = mmif if type(mmif) is Mmif else Mmif(mmif)

        # TODO: change the value of this parameter after the modelName parameter in metadata.py is solved
        model_file = '../app-gensim-tm-keywordextractor/model'
        topn = parameters['topN']

        text_slicer_checker = self.mmif.get_view_contains(DocumentTypes.TextDocument)
        new_view = self._new_view(parameters)
        if text_slicer_checker is None:
            # scenario 1: single text document input.
            for doc in self.mmif.get_documents_by_type(DocumentTypes.TextDocument):
                self._keyword_extractor(doc, new_view, doc.long_id, model_file, topn)
        else:
            # scenario 2: input document is the one generated from the text slicer.
            docs = text_slicer_checker.get_annotations(DocumentTypes.TextDocument)
            for doc in docs:
                text = doc.text_value
                if len(text.split()) > topn:
                    self._keyword_extractor(doc, new_view, doc.long_id, model_file, topn)

        # return the MMIF object
        return self.mmif

    def _new_view(self, runtime_config):
        view = self.mmif.new_view()
        view.metadata.app = self.metadata.identifier
        self.sign_view(view, runtime_config)
        view.new_contain(DocumentTypes.TextDocument, text="keywords", scores="LDA coherence scores")
        view.new_contain(AnnotationTypes.Alignment)
        return view


    def _keyword_extractor(self, doc, new_view, full_doc_id, model_file, topn):
        """Run the keyword extractor over the document and add annotations to the view, using the
        full document identifier (which may include a view identifier) for the document
        property."""
        text = doc.text_value

        # get keywords and their corresponding coherence scores for the document
        keyword_coherence_pairs = extract_keyword_coherence_pairs(model_file, text, topn=topn)
        keywords = ""
        coherence_scores = []
        for pair in keyword_coherence_pairs:
            keywords += pair[0] + " "
            coherence_scores.append(pair[1])

        # create the document to store the keywords and their tfidf values
        keywords_doc = new_view.new_textdocument(text=keywords.strip(), scores=coherence_scores)

        # create the alignment between the target document and the corresponding keywords and tfidf values
        new_view.new_annotation(AnnotationTypes.Alignment, source=full_doc_id, target=keywords_doc.long_id)


def get_app():
    """
    This function effectively creates an instance of the app class, without any arguments passed in, meaning, any
    external information such as initial app configuration should be set without using function arguments. The easiest
    way to do this is to set global variables before calling this.
    """
    return GensimTmKeywordextractor()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", action="store", default="5000", help="set port to listen")
    parser.add_argument("--production", action="store_true", help="run gunicorn server")
    # add more arguments as needed
    # parser.add_argument(more_arg...)

    parsed_args = parser.parse_args()

    # create the app instance
    # if get_app() call requires any "configurations", they should be set now as global variables
    # and referenced in the get_app() function. NOTE THAT you should not change the signature of get_app()
    app = get_app()

    http_app = Restifier(app, port=int(parsed_args.port))
    # for running the application in production mode
    if parsed_args.production:
        http_app.serve_production()
    # development mode
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()
