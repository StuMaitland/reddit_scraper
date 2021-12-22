import nltk
import pandas as pd
import re
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models
nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))
from nltk.stem import WordNetLemmatizer
stemmer = WordNetLemmatizer()


class LdaModeling():
    def __init__(self, data):
        self.df = pd.read_csv(data)
        self.corpus_superlist = self.df[['body']].values.tolist()
        #corpus_superlist
        self.corpus = []
        for sublist in self.corpus_superlist:
            for item in sublist:
                self.corpus.append(item)


    def preprocessing(self):
        def preprocess_text(document):
            # Remove all the special characters
            document = re.sub(r'\W', ' ', str(document))

            # remove all single characters
            document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

            # Remove single characters from the start
            document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

            # Substituting multiple spaces with single space
            document = re.sub(r'\s+', ' ', document, flags=re.I)

            # Removing prefixed 'b'
            document = re.sub(r'^b\s+', '', document)

            # Converting to Lowercase
            document = document.lower()

            # Lemmatization
            tokens = document.split()
            tokens = [stemmer.lemmatize(word) for word in tokens]
            tokens = [word for word in tokens if word not in en_stop]
            tokens = [word for word in tokens if len(word)  > 5]

            return tokens

        processed_data = []
        for doc in self.corpus:
            tokens = preprocess_text(doc)
            processed_data.append(tokens)

        gensim_dictionary = corpora.Dictionary(processed_data)
        gensim_corpus = [gensim_dictionary.doc2bow(token, allow_update=True) for token in processed_data]

        return gensim_corpus, gensim_dictionary

    def modeling(self):
        lda_model = gensim.models.ldamodel.LdaModel(gensim_corpus, num_topics=10, id2word=gensim_dictionary, passes=50)
        lda_model.save('gensim_model.gensim')
        return lda_model

    def plotting(self, lda_model, gensim_corpus, gensim_dictionary):
        print('display')
        return pyLDAvis.gensim_models.prepare(lda_model, gensim_corpus, gensim_dictionary)


    def performance(self, lda_model, gensim_corpus, gensim_dictionary):
        print('\nPerplexity:', lda_model.log_perplexity(gensim_corpus))
        coherence_score_lda = CoherenceModel(model=lda_model, texts=gensim_corpus, dictionary=gensim_dictionary, coherence='c_v')
        coherence_score = coherence_score_lda.get_coherence()
        print('\nCoherence Score:', coherence_score)



lda_instance = LdaModeling('2017-2021.csv')
gensim_corpus, gensim_dictionary = lda_instance.preprocessing()
lda_model = lda_instance.modeling()
lda_instance.performance(lda_model, gensim_corpus, gensim_dictionary)
lda_plot = lda_instance.plotting(lda_model, gensim_corpus, gensim_dictionary)
pyLDAvis.save_html(lda_plot, 'test.html')
print(lda_plot)

