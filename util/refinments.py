import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim.downloader as api
import numpy as np

# Load pre-trained GloVe model
model = api.load("glove-wiki-gigaword-100")


def get_synonym(word):
    """Retrieve synonyms from GloVe model based on cosine similarity."""
    try:
        # Find the top 5 most similar words
        similar_words = model.most_similar(word, topn=5)
        return [similar_word[0] for similar_word in similar_words]
    except KeyError:
        # Return the word itself if it's not in the model vocabulary
        return [word]


def refine_essay(essay, max_repetitions=2, previous_refinements=None):
    """
    Refines an IELTS essay to improve coherence, cohesion, and reduce repetition.
    Replaces words that repeat more than `max_repetitions` times with synonyms.

    Args:
        essay (str): The essay to refine.
        max_repetitions (int, optional): The maximum number of repetitions before replacing a word. Defaults to 2.
        previous_refinements (list, optional): A list of previous refinements to avoid repetition. Defaults to None.

    Returns:
        str: The refined essay.
    """

    sentences = sent_tokenize(essay)
    stop_words = set(stopwords.words("english"))

    # Track word frequency
    word_freq = {}
    for sentence in sentences:
        words = [word.lower() for word in word_tokenize(sentence) if word.isalnum()]
        for word in words:
            if word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1

    # Replace words that repeat more than the allowed threshold
    refined_sentences = []
    if previous_refinements is None:
        previous_refinements = []

    for i, sentence in enumerate(sentences):
        words = [word.lower() for word in word_tokenize(sentence) if word.isalnum()]
        refined_words = []

        for word in words:
            if word_freq.get(word, 0) > max_repetitions:
                # Replace word with a synonym if it repeats more than allowed
                synonyms = get_synonym(word)
                refined_words.append(
                    np.random.choice(synonyms)
                )  # Randomly choose a synonym for variety
            else:
                refined_words.append(word)

        refined_sentence = " ".join(refined_words)
        refined_sentences.append(
            refined_sentence.capitalize() + "."
        )  # Capitalize and add period

        previous_refinements.append(sentence)  # Track refinement

    return " ".join(refined_sentences)


# Example usage
nltk.download("punkt")
nltk.download("stopwords")

essay = "This is an important essay. Information technology is very important. Important things are important. This essay is also important."
print(essay)
refined_essay_text = refine_essay(essay)
print("Refined Essay:")
print(refined_essay_text)

essay = "This is an important essay. Information technology is very important. Important things are important. This essay is also important."
print(essay)
refined_essay_text = refine_essay(essay)
print("Refined Essay:")
print(refined_essay_text)
