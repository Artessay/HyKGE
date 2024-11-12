import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

class EntityLinker():
    def __init__(self, encode_method: str = 'GTE') -> None:
        assert encode_method in ['GTE']
        dir_path = os.path.dirname(os.path.abspath(__file__))

        # self.entity_columns = ['entity_id', 'entity_name', 'entity_type']
        self._entity_dataset = pd.read_pickle(f'{dir_path}/resource/entity.pkl')

        encode_method = encode_method.lower()
        self._entity_embeddings = EntityLinker._load_normalized_embeddings(f'{dir_path}/resource/embeddings_{encode_method}.npy')
        self._encoding_model = SentenceTransformer(f'{dir_path}/resource/checkpoint_{encode_method}/')

    def entity_linking(self, mention_list: list, sim_threshold: float = 0.6) -> list:
        # deal with None value
        if mention_list is None or mention_list == []:
            return []
        
        # encode and normalize mention embeddings
        mention_embeddings = self._calculate_mention_embedding(mention_list)
        mention_embeddings = EntityLinker._normalize_and_stack_list_of_vectors(mention_embeddings)
        
        # extract entity indices for these mentions
        entity_indices = self._extract_entity_indices(mention_embeddings, sim_threshold)
        entity_indices = list(set(entity_indices))

        # map entity indices to entity names
        entity_list = [self._entity_dataset['entity_name'][entity_index] for entity_index in entity_indices]

        return entity_list
    
    def _calculate_mention_embedding(self, mention_list):
        embedding_list = [
            self._encoding_model.encode(value)
            for value in mention_list
        ]

        return embedding_list

    def _extract_entity_indices(self, mention_embeddings: np.array, sim_threshold: float):
        entity_embeddings: np.array = self._entity_embeddings
        inner_product_matrix = np.dot(mention_embeddings, entity_embeddings.T)
        
        entity_index_list = []
        for i in range(len(mention_embeddings)):
            max_index = np.argmax(inner_product_matrix[i])
            similarity = inner_product_matrix[i][max_index]
            # We only select entities whose scores are higher than threshold
            if similarity >= sim_threshold:
                entity_index_list.append(max_index)

        return entity_index_list
    
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
        return np.stack([EntityLinker._normalize_vector(vector) for vector in list_of_vectors])


if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    mention_list = ['破伤风', '青霉素', '铁锈挂伤']

    entity_linker = EntityLinker()
    entity_list = entity_linker.entity_linking(mention_list)
    entity_list = sorted(entity_list)
    print(entity_list)

    assert entity_list == ['破伤风', '青霉素']