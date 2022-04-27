import numpy as np
import pandas as pd
import PyPDF2
import torch
import time
import googletrans as gt
from tqdm import tqdm
from gensim.models import Word2Vec
from joblib import load, dump

def read_raw_pdf(path, verbose=0):
    import PyPDF2
    try:
        pdfReader = PyPDF2.PdfFileReader(path)
        #Discerning the number of pages will allow us to parse through all the pages.
        num_pages = pdfReader.numPages
    except:
        if verbose > 1:
            print(f'Error de lectura de PDF en: {path}')
        raise
    count = 0
    pages = []
    if verbose > 1: print(f'\n{path.split("/")[-1]}')
    while count < num_pages:
        if verbose > 1: print(f'Page {count+1}/{num_pages}', end="\r")
        pageObj = pdfReader.getPage(count)
        pages.append(pageObj.extractText())
        count += 1

    pdf_dict = {
        'path':path,
        'pages':pages,
    }
    return pdf_dict

def prepare_PDF_dataframe(raw_texts):
    import pandas as pd
    data = pd.DataFrame(raw_texts)
    data['bank'] = data['path'].apply(lambda x: x.split('/')[-1].split('_')[0])
    data['year'] = data['path'].apply(lambda x: x[-8:-4])
    data = data.explode('pages').reset_index().reset_index()
    data['page'] = data.groupby('index').rank('max').rename(columns={'level_0':'page'})['page']
    data['page'] = data['page'].astype('int')
    data = (data.drop(columns=['level_0','index','path'])
               .set_index(['bank','year','page'])
               .rename(columns={'pages':'text'})
          )
    return data

def clean_text(text, esp=False):
    import re
    text = re.sub(r"\n\n", ".", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(":", " ", text)
    text = re.sub(r"[^a-zA-Z.\s:ñÑáéíóúÁÉÍÓÚ]", "", text)
    text = re.sub(r' +', ' ', text)
    text = text.strip()
    if text == '':
        return None
    if esp:
        try:
            text=translate_ES(text)
        except:
            return None
    return list(filter(None, text.split('.')))

def tokenize_phrases(df_data, esp=False):
    tqdm.pandas()
    if esp: print('--- Realizando traducción al inglés...')
    df_data['sentences'] = df_data['text'].progress_apply(lambda x: clean_text(x, esp=esp))
    df_data = df_data.dropna()
    df_sentence = df_data.explode('sentences').dropna()
    df_sentence['sentence'] = df_sentence.groupby(df_sentence.index).rank('max')['sentences']
    df_sentence['sentence'] = df_sentence['sentence'].astype('int')
    df_sentence.set_index('sentence', append=True, inplace=True)
    df_sentence.drop(columns=['text'], inplace=True)
    return df_sentence

translator = gt.Translator()

def translate_ES(sentence):
    return translator.translate(sentence, dest='en', src='es').text

def process_PDFs(list_of_paths,
                 timeit=True,
                 verbose=0,
                 esp=False,
                 save_path=None,
                 save_raw_text=None):
    import time
    total_times = dict()
    if timeit:
        print('Cronometrando lectura de PDFs...')
        start_time = time.time()

    raw_texts = []
    for path in list_of_paths:
        try:
            pdf_dict = read_raw_pdf(path, verbose=verbose)
        except:
            continue
        raw_texts.append(pdf_dict)

    if timeit:
        end_time = time.time()
        total_times["lectura"] = end_time - start_time
        if verbose: print(f'--- Tiempo de lectura (s): {end_time - start_time}')

    # Textos a DataFrame:
    if verbose: print(f'Preparando dataframe...')
    raw_text_data = prepare_PDF_dataframe(raw_texts)

    # if timeit:
    #     end_time = time.time()
    #     total_times["creacion_dataframe"] = end_time - total_times["lectura"]

    # Guardado de texto RAW:
    if save_raw_text:
        raw_text_data.to_parquet(save_raw_text)

    if verbose: print(f'Tokenizando frases...')
    df_phrases = tokenize_phrases(raw_text_data, esp=esp)

    # Guardando dataset:
    if save_path:
        print('Guardando dataset...')
        df_phrases.to_parquet(save_path)
        print('Dataset guardado!')

    return df_phrases, total_times

def filter_phrases_size(df_sentence, min_words=10, tensor_limit=512):
    df_sentence['length'] = df_sentence['sentences'].apply(lambda x: len(x.strip().split(' ')))
    return df_sentence[(df_sentence.length >= min_words) & (df_sentence.length <= tensor_limit)]

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1)[:, None])
    return e_x / np.sum(e_x, axis=1)[:, None]

def sentimentAnalysis_full(text_payload, tokenizer, model, tensor_limit=512):
    inputs = tokenizer(text_payload, return_tensors="pt")["input_ids"][:,:tensor_limit]
    logits = model(inputs).logits
    return softmax(np.array(logits.detach()))[0]

def get_phrase_sentiment(dataset, min_words=10, tensor_limit=512, save_sentiment_path=None, timeit=True):
    import time
    import pandas as pd
    from tqdm import tqdm
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    MODEL_PATH = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    pd.options.mode.chained_assignment = None #'warn'
    tqdm.pandas()
    total_times = dict()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    if timeit:
        print('Cronometrando procesado de sentiment de frases...')
        start_sentiment_time = time.time()

    dataset = filter_phrases_size(dataset, min_words=min_words, tensor_limit=tensor_limit)
    dataset['sentiment'] = dataset.sentences.progress_apply(
        lambda x: sentimentAnalysis_full(x, tokenizer, model, tensor_limit=tensor_limit)
    )

    if timeit:
        end_sentiment_time = time.time()
        total_times['sentiment'] = end_sentiment_time - start_sentiment_time

    dataset['negative'] = dataset['sentiment'].progress_apply(lambda x: x[0])
    dataset['neutral']  = dataset['sentiment'].progress_apply(lambda x: x[1])
    dataset['positive'] = dataset['sentiment'].progress_apply(lambda x: x[2])
    df_export = dataset.drop(columns='sentiment')

    if save_sentiment_path:
        df_export.to_parquet(save_sentiment_path)

    return df_export, total_times

def refactor_data(data):
    return (data
            .reset_index()
            .rename(columns={'sentences':'phrase', 'page':'n_page', 'sentence':'n_phrase'},)
            .set_index(['bank','year', 'n_page', 'n_phrase']))

def get_top_positive_phrases(data, count=20):
    data = refactor_data(data)
    preview = data.sort_values(by='positive', ascending=False).head(count)[['phrase']]
    return preview

def get_top_negative_phrases(data, count=20):
    data = refactor_data(data)
    preview = data.sort_values(by='negative', ascending=False).head(count)[['phrase']]
    return preview

def palabra_en_modelo(palabra,w2v):
    try:
        w2v.wv.get_vector(palabra)
        return True
    except:
        return False

def vectorize_text(texto_vector, w2v, SIZE_VECTORS=100):
    # Complete here the code
    vectors = [w2v.wv.get_vector(i) for i in texto_vector if palabra_en_modelo(i,w2v)]
    return np.mean(vectors, axis=0) if len(vectors) else np.zeros(SIZE_VECTORS)

def prepare_dataset(data, path_word2vec, path_clustering, SIZE_VECTORS=100):
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    tqdm.pandas()
    data = refactor_data(data)

    word2vec = Word2Vec.load(path_word2vec)
    print('Tokenizing...')
    data['texto_vector'] = data.phrase.progress_apply(lambda x: [w.lower() for w in x.split(' ') if w])
    print('Aplicando Word2Vec...')
    vectors = data.texto_vector.progress_apply(lambda x: vectorize_text(x, word2vec, SIZE_VECTORS)).to_numpy()

    dfVectors = pd.DataFrame(np.concatenate(vectors).reshape(vectors.shape[0],SIZE_VECTORS,), index=data.index,
                         columns=[f'WV{i}' for i in range(SIZE_VECTORS)])
    print('Aplicando Clustering...')
    clustering = load(path_clustering)
    dfVectors = np.array(dfVectors, dtype=np.double)
    clusters = clustering.predict(dfVectors)
    data['cluster'] = clusters
    data = data.dropna()
    print('Preparando Dataset...')
    ml_df = data.reset_index().groupby(['bank','year','cluster']).mean()[['positive','negative']].unstack('cluster')
    ml_df.columns = [sent[:3]+'_cluster_'+str(cluster)  for sent, cluster in ml_df.columns.values]
    ml_df = ml_df.fillna(0.0)

    return ml_df

def get_mergers(ml_df, path_merger_data):
    df_mergers = pd.read_csv(path_merger_data).dropna()
    df_mergers['YEAR'] = df_mergers['YEAR'].astype('int')
    df_mergers = df_mergers.set_index('YEAR')

    merger_target = []
    for bank, year in ml_df.index:
        merger_target.append(int(df_mergers[bank][int(year)]))

    return merger_target

def predict_merger(data,
                   path_modelo_word2vec,
                   path_modelo_clustering,
                   path_modelo_merger,
                   timeit=True
                  ):
    import time
    from joblib import load
    total_time = 0
    if timeit:
        start = time.time()
    X = prepare_dataset(data, path_modelo_word2vec, path_modelo_clustering)
    # Load del modelo:
    print('Cargando Modelo...')
    model = load(path_modelo_merger)
    print('Haciendo Predicción...')
    merger_pred = model.predict_proba(X)[:,1]
    if timeit:
        end = time.time()
        total_time = end - start

    # Refactor Dataset de salida
    new_cols = [(col.split('_')[0],'cluster_'+col.split('_')[-1])  for col in X.columns]
    X.columns = pd.MultiIndex.from_tuples(new_cols)

    return merger_pred, {'prediction':total_time}, X.stack()
