import spacy
from collections import Counter
from nltk.corpus import stopwords
import string
import re
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

#Portuguese language model
nlp = spacy.load("pt_core_news_sm")

#List of Portuguese stopwords
stop_words = set(stopwords.words("portuguese"))

#Output directory for generated charts and CSV files
base_output_dir = Path("C:/Users/ruypa/OneDrive/Ambiente de Trabalho/MHD/2o Semestre/Análise e Visualização de Dados/Rui/extractions")

def read_file(filename):
    with open(filename, "r", encoding="utf-8") as file:
        text = file.read()
    return text

def clean_text(text):
    #Removes newlines and extra whitespaces
    text = re.sub("\s+", " ", text)
    #Removes punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    #Converts to lowercase
    text = text.lower()
    #Removes stopwords
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

def extract_people(doc):
    return [entity.text for entity in doc.ents if entity.label_ == "PER"]

def extract_places(doc):
    return [entity.text for entity in doc.ents if entity.label_ == "LOC"]

def extract_lemmas(text):
    tokens = nltk.word_tokenize(text, language='portuguese')
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token.lower() not in stop_words]
    return lemmas

def extract_keywords(doc):
    keywords = []
    for token in doc:
        if token.is_alpha and not token.is_stop and token.pos_ in ["NOUN", "ADJ"]:
            keywords.append(token.text)
            for child in token.children:
                if child.pos_ == "ADJ":
                    keywords.append(child.text)
    return keywords

def sentiment_analysis(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    return sentiment

def export_to_csv(data, filename):
    df = pd.DataFrame(data, columns=["Extracted Data", "Frequency"])
    df.to_csv(filename, index=False)

def create_bar_plot(data, title, xlabel, ylabel, filename):
    entities = [entity for entity, count in data]
    counts = [count for entity, count in data]

    plt.figure(figsize=(10, 6))
    plt.bar(entities, counts)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def analyze_file(filename, output_dir):
    text = read_file(filename)
    cleaned_text = clean_text(text)
    doc = nlp(cleaned_text)

    people = extract_people(doc)  #Extracts people
    places = extract_places(doc)  #Extracts places
    lemmas = extract_lemmas(cleaned_text)  #Extracts lemmas
    keywords = extract_keywords(doc)  #Extracts keywords

    #Performs sentiment analysis on the cleaned text
    sentiment = sentiment_analysis(cleaned_text)

    #Counts the frequency of each entity
    people_count = Counter(people).most_common(10)
    places_count = Counter(places).most_common(10)
    lemmas_count = Counter(lemmas).most_common(10)
    keywords_count = Counter(keywords).most_common(10)

    #Creates the output directory for this file
    output_dir = base_output_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    #Exports the entity counts to CSV files
    export_to_csv(people_count, output_dir / "people.csv")
    export_to_csv(places_count, output_dir / "places.csv")
    export_to_csv(lemmas_count, output_dir / "lemmas.csv")
    export_to_csv(keywords_count, output_dir / "keywords.csv")

    #Creates bar plots for entity counts
    create_bar_plot(people_count, f"Top 10 People in {file_mappings[file]}", "Person", "Count", output_dir / "people.png")
    create_bar_plot(places_count, f"Top 10 Places in {file_mappings[file]}", "Place", "Count", output_dir / "places.png")
    create_bar_plot(lemmas_count, f"Top 10 Lemmas in {file_mappings[file]}", "Lemma", "Count", output_dir / "lemmas.png")
    create_bar_plot(keywords_count, f"Top 10 Keywords in {file_mappings[file]}", "Keyword", "Count", output_dir / "keywords.png")

    #Creates a pie chart for sentiment scores
    labels = ['Negative', 'Neutral', 'Positive']
    sizes = [sentiment['neg'], sentiment['neu'], sentiment['pos']]
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    explode = (0.1, 0, 0)  #explode the 'Negative' slice

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, colors=colors, explode=explode, autopct='%1.1f%%', startangle=90)
    plt.title(f"Sentiment Distribution in {file_mappings[file]}")
    plt.axis('equal')
    plt.savefig(output_dir / "sentiment_pie.png")
    plt.close()

#Analyze each file
file_mappings = {
    "C:/Users/ruypa/OneDrive/Ambiente de Trabalho/MHD/2o Semestre/Análise e Visualização de Dados/Rui/Obra/Camilo-A_Infanta_Capelista.txt":
        "Camilo-A_Infanta_Capelista",
    "C:/Users/ruypa/OneDrive/Ambiente de Trabalho/MHD/2o Semestre/Análise e Visualização de Dados/Rui/Obra/Camilo-A_viuva_do_enforcado.txt":
        "Camilo-A_viuva_do_enforcado",
    "C:/Users/ruypa/OneDrive/Ambiente de Trabalho/MHD/2o Semestre/Análise e Visualização de Dados/Rui/Obra/Camilo-O_carrasco_de_Vitor_Hugo.txt":
        "Camilo-O_carrasco_de_Vitor_Hugo",
    "C:/Users/ruypa/OneDrive/Ambiente de Trabalho/MHD/2o Semestre/Análise e Visualização de Dados/Rui/Obra/Camilo-Vulcoes_de_lama.txt":
        "Camilo-Vulcoes_de_lama"
}

for file, output_dir in file_mappings.items():
    analyze_file(file, output_dir)

print("|--------------------------------------------------------------------------------------------|")
print("|Extraction completed successfully. Please check the folder \"extractions\" to access the data.|")
print("|--------------------------------------------------------------------------------------------|")
