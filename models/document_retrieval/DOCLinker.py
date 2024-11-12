import os
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

class DocumentLinker():
    def __init__(self, encode_method: str = 'GTE') -> None:
        assert encode_method in ['GTE', 'Piccolo']     # GTE or Piccolo
        dir_path = os.path.dirname(os.path.abspath(__file__))

        # self.entity_columns = ['entity_id', 'entity_name', 'entity_type']

        encode_method = encode_method.lower()
        self._entity_dataset = pd.read_pickle(f'{dir_path}/resource/others/document_{encode_method}_tjy.pkl')
        self._faiss_index = faiss.read_index(f'{dir_path}/resource/others/faiss_index_{encode_method}_tjy.bin')

        self._encoding_model = SentenceTransformer(f'{dir_path}/resource/checkpoint_{encode_method}/')

    def entity_linking(self, query_list: list, sim_threshold: float = 0.6) -> list:
        # deal with None value
        if query_list is None or query_list == []:
            return []

        # encode and normalize query embeddings
        query_embeddings = self._encoding_model.encode([query_list], normalize_embeddings=True)
        # query_embeddings = self._calculate_mention_embedding(query_list)
        # query_embeddings = DocumentLinker._normalize_and_stack_list_of_vectors(query_embeddings)

        # extract entity indices for these mentions
        entity_list = self._search(query_embeddings, sim_threshold)
        entity_list = list(set(entity_list))

        return entity_list

    def _search(self, query_embedding, sim_threshold=0, TopK=5):
        # 使用FAISS索引进行查询
        distances, indices = self._faiss_index.search(query_embedding, TopK)

        # 获取最相似的句子及其相似度
        similar_sentences = []
        for distance, idx in zip(distances[0], indices[0]):
            # if 1 - distance >= sim_threshold:
            print(idx)
            similar_sentences.append(self._entity_dataset[idx])

        return similar_sentences

    def _calculate_mention_embedding(self, mention_list):
        embedding_list = [
            self._encoding_model.encode(value)
            for value in mention_list
        ]

        return embedding_list

    @staticmethod
    def _load_normalized_embeddings(embedding_path):
        # load embeddings
        embeddings = np.load(embedding_path)

        # normalize embeddings
        row_normals = np.linalg.norm(embeddings, axis=1)
        normalized_embeddings = embeddings / row_normals[:, np.newaxis]

        return normalized_embeddings

    @staticmethod
    def _normalize_vector(vector):
        norm = np.linalg.norm(vector)
        return vector / norm if norm != 0 else vector

    @staticmethod
    def _normalize_and_stack_list_of_vectors(list_of_vectors):
        return np.stack([DocumentLinker._normalize_vector(vector) for vector in list_of_vectors])




if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    mention_list = ['破伤风', '青霉素', '铁锈挂伤']

    entity_linker = DocumentLinker()
    entity_list = entity_linker.entity_linking(mention_list)
    entity_list = sorted(entity_list)
    print(entity_list)

    assert entity_list == ['破伤风', '青霉素']