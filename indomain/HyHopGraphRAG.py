from retrieval_tools import *
from embedding_model import *
from prompt import *
from utils import *
from llm import *

import itertools
import concurrent.futures

class EnhancedReasoner:
    def __init__(self):
        self.chunk_retriever = ChunkRetriever()
        self.graph_retriever = GraphRetriever()

        self.generator = Generator()
        self.graph_db = self.GraphDB_Init()

    def GraphDB_Init(self):
        from langchain_community.graphs import Neo4jGraph
        from src.neo4j_loader import Neo4jLoader

        config = {
            "url": "neo4j://xxx",
            "username": "neo4j",
            "password": "xxx",
            "database": "xx"
        }
        graph_db = Neo4jGraph(
            url=config['url'],
            username=config['username'],
            password=config['password'],
            database=config['database']
        )

        loader = Neo4jLoader(config)

        # Load all documents
        # loader.load_all("results")

#         loader.resolve_duplicate_labels()
#         loader.merge_generic_entity_nodes()
#         # loader.create_parent_relationships()
#         loader.merge_misidentified_documents()

        return graph_db

    def fetch_paths(self, entity_from, entity_to, graph_db, graph_retriever):
        # if entity_from == entity_to:
        #     return []
        # else:
        return graph_retriever.search_paths_from_entities(graph_db, from_entity=entity_from, to_entity=entity_to)

    def process_paths(self, flattened_entities, graph_db, graph_retriever):
        # Step: 使用线程池并行获取路径
        all_paths = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # 生成所有 entity_from 和 entity_to 组合的任务
            futures = [
                executor.submit(self.fetch_paths, entity_from, entity_to, graph_db, graph_retriever)
                for entity_from in flattened_entities
                for entity_to in flattened_entities
                if entity_from != entity_to
            ]
            
            for future in concurrent.futures.as_completed(futures):
                paths = future.result()
                if paths is not None: 
                    all_paths.append(paths)

        return all_paths

    def reason(self, user_query: str) -> dict:
        ## 1. todo Use Cypher to Search Chunks and Obtain topK (3) descriptions
        ## in-domain knowledge
        text_chunks = ChunkRetriever.recall_docs_text(self, graph_db=self.graph_db, user_query=user_query, limit=3)

        ### 2. Then concat descriptions to LLMs and obtain [1] Hypothesis Output [2] Conquar&Solve
        pre_prompt_HO = HypothesisOutput(user_query, text_chunks)
        pre_prompt_CAS = ConquerAndSolve(user_query, text_chunks)

        pre_response_HO = self.generator.generate(pre_prompt_HO)
        pre_response_CAS = self.generator.generate(pre_prompt_CAS)

        pre_response = pre_response_HO + pre_response_CAS
        print("pre response:", pre_response)
        ### 2.1 Chunk the pre_response via chunksize = 40
        chunks = chunk_text_by_words(user_query+pre_response)

        ### 3. todo Use Cypher to obtain entity list via entity searching for each chunk
        # Example usage:
        all_entities = []
        # print(len(chunks))
        # for chunk in chunks:
        ## todo: test first chunk
        for chunk in chunks[:1]:
            chunk_embedding = get_embeddings_batch(chunk)
            entities_in_chunk = self.graph_retriever.extract_entities_from_chunk(graph_db=self.graph_db, input_embedding=chunk_embedding, topK=10)
            all_entities.append(entities_in_chunk)
        flattened_entities = list(itertools.chain(*all_entities))
        flattened_entities = list({d["node"].get("id", d["node"].get("name")) for d in flattened_entities})
        print("entities:", flattened_entities)

        ### 4. Obtain paths => 并行优化
        all_paths = self.process_paths(flattened_entities, self.graph_db, self.graph_retriever)

        flattened_paths = list(itertools.chain(*all_paths))
        # flat all nodes
        all_nodes = list(itertools.chain(*flattened_paths))
        flattened_paths = list({node["id"] for node in all_nodes if isinstance(node, dict) and "id" in node})

        ### 4.5 Reranker
        reranked_flattened_paths = reranker_chains(user_query, flattened_paths)

        ### 5. Organize path knowledge
        middle_prompt = KnowledgeIntegrationPrompt(user_query, reranked_flattened_paths)
        retrieved_knowledege = self.generator.generate(middle_prompt)

        ### 6. To LLM and obtain results
        final_prompt = RAGprompt(user_query, retrieved_knowledege)
        final_response = self.generator.generate(final_prompt)
        print("%%%%%%%%%%%% \n LLM Final Answer:", final_response)

        # 返回包括子图、supporting facts 及最终答案的结果
        return {
            # "subgraph": subgraph,
            # "supporting_facts": supporting_facts,
            "user query": user_query,
            "entities": flattened_entities,
            "paths": reranked_flattened_paths,
            "final_response": final_response
        }


def main(query):
    reasoner = EnhancedReasoner()
    result = reasoner.reason(query)
    print("最终返回结果：", result)


if __name__ == '__main__':
    query = "What are the top 5 topics in the corpus?"

    main(query)