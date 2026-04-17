import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

class TextPreprocessor:
    """
    Pipeline de nettoyage des synopsis :
    - lowercase
    - suppression ponctuation / chiffres / caractères spéciaux
    - tokenisation
    - suppression stopwords
    - lemmatisation OU stemming (au choix)
    """
    
    def __init__(self, use_lemmatization=True):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.use_lemmatization = use_lemmatization

    def clean(self, text: str) -> str:
        # 1. Lowercase
        text = text.lower()
        # 2. Supprimer les balises HTML éventuelles
        text = re.sub(r'<[^>]+>', '', text)
        # 3. Supprimer les chiffres et ponctuations
        text = re.sub(r'[^a-z\s]', '', text)
        # 4. Supprimer les espaces multiples
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize(self, text: str) -> list[str]:
        tokens = word_tokenize(text)
        # Supprimer stopwords + tokens trop courts
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        # Lemmatisation ou stemming
        if self.use_lemmatization:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        else:
            tokens = [self.stemmer.stem(t) for t in tokens]
        return tokens

    def preprocess(self, text: str) -> str:
        """Retourne le texte nettoyé sous forme de string (pour TF-IDF/BERT)."""
        cleaned = self.clean(text)
        tokens = self.tokenize(cleaned)
        return ' '.join(tokens)

    def preprocess_to_tokens(self, text: str) -> list[str]:
        """Retourne une liste de tokens (pour Word2Vec)."""
        cleaned = self.clean(text)
        return self.tokenize(cleaned)

    def preprocess_batch(self, texts: list[str]) -> list[str]:
        return [self.preprocess(t) for t in texts]

    def preprocess_batch_tokens(self, texts: list[str]) -> list[list[str]]:
        return [self.preprocess_to_tokens(t) for t in texts]