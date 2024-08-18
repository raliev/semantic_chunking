import re
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances, pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from copy import deepcopy
import spacy
import sys
from sentence_transformers import SentenceTransformer
import logging
#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#logger = logging.getLogger(__name__)

class TextChunker:
    def __init__(self, max_recursion_depth=5, aggressiveness=5):
        self.SMOOTHNESS_ON = 1
        self.MAX_SMOOTHNESS = 2
        self.LOOK_FORWARD=1
        self.LOOK_BACKWARD=5
        #euclidian, cosine_similarity, manhattan, minkowski
        self.COMPARE_STRATEGY = "cosine_similarity"
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Fine-tuned model for semantic similarity
        self.max_recursion_depth = max_recursion_depth
        self.aggressiveness = aggressiveness  # Parameter for controlling the aggressiveness of the splitting
        self.nlp = spacy.load('en_core_web_sm')

    # Splitting text into sentences
    def _split_sentences(self, text):
        doc = self.nlp(text)
        sentences = [sent.text for sent in doc.sents]
        return sentences

    # Combining sentences using a buffer
    def _combine_sentences(self, sentences, buffer_size):
        for i in range(len(sentences)):
            combined_sentence = ''
            for j in range(i - buffer_size, i):
                if j >= 0:
                    combined_sentence += str(sentences[j]['sentence']) + ' '
            combined_sentence += str(sentences[i]['sentence'])
            for j in range(i + 1, i + 1 + buffer_size):
                if j < len(sentences):
                    combined_sentence += ' ' + str(sentences[j]['sentence'])
            sentences[i]['combined_sentence'] = combined_sentence
        return sentences

    def _smooth_distances(self, distances, window_size):
        smoothed_distances = np.convolve(distances, np.ones(window_size) / window_size, mode='same')
        return smoothed_distances.tolist()

    # Calculating cosine distances between combined sentences
    # Calculating cosine distances between combined sentences
    def _calculate_cosine_distances(self, sentences):
        enriched_sentences = deepcopy(sentences)
        distances = []
        updated = [0] * len(enriched_sentences)

        for i in range(len(enriched_sentences) - 1):
            if enriched_sentences[i].get("distance", None) is not None:
                distances.append(enriched_sentences[i].get("distance"))
                updated[i] = 0
                continue

            # Используем эмбеддинги N_back и N_forward
            embedding_current_back = enriched_sentences[i]['combined_sentence_embedding_N_back']
            embedding_next_forward = enriched_sentences[i]['combined_sentence_embedding_N_forward']

            distance = 0;
            # Вычисляем косинусное сходство между embedding_current_back и embedding_next_forward
            distance_metric = self.COMPARE_STRATEGY
            if distance_metric == 'cosine_similarity':
                similarity = cosine_similarity([embedding_current_back], [embedding_next_forward])[0][0]
                distance = 1 - similarity
            elif distance_metric == 'euclidean':
                distance = euclidean_distances([embedding_current_back], [embedding_next_forward])[0][0]
            elif distance_metric == 'manhattan':
                distance = manhattan_distances([embedding_current_back], [embedding_next_forward])[0][0]
            elif distance_metric == 'minkowski':
                distance = pairwise_distances([embedding_current_back], [embedding_next_forward], metric='minkowski', p=3)[0][0]
            #elif distance_metric == 'kl_divergence':
            #    distance = entropy(embedding_current_back, embedding_next_forward)
            #elif distance_metric == 'hellinger':
            #    distance = hellinger(embedding_current_back, embedding_next_forward)
            else:
                raise ValueError(f"Unknown distance metric: {distance_metric}")


            distances.append(distance)
            updated[i] = 1

        # Apply smoothing to the distances array and convert it back to a list
        if distances and self.SMOOTHNESS_ON:
            window_size = max(self.MAX_SMOOTHNESS, self.aggressiveness)  # Adjust window size based on aggressiveness
            distances = self._smooth_distances(distances, window_size)

        for i in range(len(enriched_sentences) - 1):
            if updated[i] == 1:
                enriched_sentences[i]['distance'] = distances[i]

        return distances, enriched_sentences


    # Grouping sentences based on split points
    def _create_groups(self, sentences, breakpoint_distance_threshold):
        distances, sentences_with_distances = self._calculate_cosine_distances(sentences)
        if not distances:
            return [sentences]

        indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold]

        groups = []
        start_index = 0
        for index in indices_above_thresh:
            group = sentences[start_index:index]
            groups.append(group)
            start_index = index

        if start_index < len(sentences):
            group = sentences[start_index:]
            groups.append(group)

        # Ensure there are at least two groups (if fewer, combine small groups)
        if len(groups) < 2:
            midpoint = len(sentences) // 2
            groups = [sentences[:midpoint], sentences[midpoint:]]

        return groups

    # Recursive function for splitting text into hierarchical groups
    def _recursive_chunking(self, sentences, level=1):
        if len(sentences) <= 1 or level > self.max_recursion_depth:
            if len(sentences) > 0:
                return {'chunk': ' '.join([s['sentence'] for s in sentences]), 'cluster' : sentences[0].get('cluster', None), 'subchunks': [], 'distance': sentences[0].get('distance', None)}
            else:
                return {}

        # Dynamic calculation of the split threshold based on aggressiveness
        distances, sentences_with_distances = self._calculate_cosine_distances(sentences)

        # Используем правильную проверку для пустого массива distances
        if len(distances) == 0:  # Или distances.size == 0 для NumPy массива
            return {'chunk': sentences[0]['sentence'], 'subchunks': [], 'distance': sentences_with_distances[0].get('distance', None)}

        # The higher the aggressiveness, the lower the percentile threshold, making the split more aggressive
        breakpoint_percentile_threshold = 90 - self.aggressiveness * 5  # Adjust this multiplier based on the desired range
        breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold)

        # Buffer size adjustment based on aggressiveness
        buffer_size = max(1, 5 - self.aggressiveness // 2)  # Adjust buffer size based on aggressiveness

        # Combining sentences with a specified buffer size
        combined_sentences = self._combine_sentences(sentences_with_distances, buffer_size)

        # Grouping sentences into clusters
        groups = self._create_groups(combined_sentences, breakpoint_distance_threshold)

        # Recursive splitting of each group and combining them into larger groups at higher levels
        subchunks = [self._recursive_chunking(group, level + 1) for group in groups]

        # Combining groups into a larger chunk at the current level
        combined_text = ' '.join([s['sentence'] for s in combined_sentences])

        dtn = sentences[0].get('distance', None)

        return {
            'chunk': combined_text,
            'subchunks': subchunks,
            'distance': dtn,
            'cluster' : sentences[0].get('cluster', None)
        }

    # Main function for splitting text into hierarchical parts
    def chunk_text(self, text):
        single_sentences_list = self._split_sentences(text)
        sentences = [{'sentence': sentence} for sentence in single_sentences_list]

        # Initializing embeddings using Sentence-BERT
        embeddings = self.model.encode([x['sentence'] for x in sentences])

        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=5)
        cluster_labels = kmeans.fit_predict(embeddings)
        for i, sentence in enumerate(sentences):
            sentence['cluster'] = str(cluster_labels[i])

        # Assigning embeddings to sentences and calculating combined embeddings
        for i, sentence in enumerate(sentences):
            sentence['combined_sentence_embedding'] = embeddings[i]

            # Calculate combined_sentence_embedding_N_back
            back_sentences = single_sentences_list[max(0, i - self.LOOK_BACKWARD + 1):i+1]
            sentence['combined_sentence_embedding_N_back'] = self.model.encode(' '.join(back_sentences))

            # Calculate combined_sentence_embedding_N_forward
            if i+1 != len(single_sentences_list):
                forward_sentences = single_sentences_list[i+1:min(len(single_sentences_list), i + self.LOOK_FORWARD + 1)]
            else:
                forward_sentences = [''];
            sentence['combined_sentence_embedding_N_forward'] = self.model.encode(' '.join(forward_sentences))

        # Recursive splitting
        hierarchical_structure = self._recursive_chunking(sentences)

        return hierarchical_structure

def _remove_embeddings_combined_sentences(structure):
    if isinstance(structure, dict):
        if 'combined_sentence_embedding' in structure:
            del structure['combined_sentence_embedding']
        if 'combined_sentence_embedding_N_back' in structure:
            del structure['combined_sentence_embedding_N_back']
        if 'combined_sentence_embedding_N_forward' in structure:
            del structure['combined_sentence_embedding_N_forward']
        if 'subchunks' in structure:
            for subchunk in structure['subchunks']:
                _remove_embeddings_combined_sentences(subchunk)
    elif isinstance(structure, list):
        for item in structure:
            _remove_embeddings_combined_sentences(item)

# Example usage
with open(sys.argv[1]) as f:
    text = f.read()
    chunker = TextChunker(max_recursion_depth=9, aggressiveness=3)  # You can change the aggressiveness level and N
    hierarchical_structure = chunker.chunk_text(text)
    _remove_embeddings_combined_sentences(hierarchical_structure)

    import json
    print(json.dumps(hierarchical_structure, indent=4))
