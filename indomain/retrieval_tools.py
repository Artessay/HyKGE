class ChunkRetriever:
    def recall_docs_text(self, graph_db, user_query: str, limit: int = 10) -> list:
        """
        find topK section source_text based on user_query
        """
        cypher = (
            "CALL db.index.fulltext.queryNodes('sectionSourceTextIndex', $user_query) "
            "YIELD node, score "
            "RETURN node.source_text AS text, score "
            "ORDER BY score DESC "
            "LIMIT $limit"
        )
        params = {"user_query": user_query, "limit": limit}
        try:
            result = graph_db.query(cypher, params=params)

            texts = [item["text"] for item in result]

            return texts if result else []
        
        except Exception as e:
            print("ChunkRetriever error: ", e)
            return []
    
    def rerank_docs(self, docs: list) -> list:
        """
        already sorted
        """
        return docs
    

# ------------------------------
# 2. GraphRetriever create subgraph
# ------------------------------
class GraphRetriever:
    
    def extract_entities_from_chunk(self, graph_db, input_embedding, topK: int) -> dict:
        ### TODO： 根据输入的embedding，利用相似度匹配，找到对应的实体
        cypher = (
            "MATCH (n) "
            "WITH n, "
            "    CASE "
            "        WHEN n.name_de_embedding IS NOT NULL THEN [n.name_de_embedding] "
            "        WHEN n.name_embedding IS NOT NULL THEN [n.name_embedding] "
            "        ELSE [propName IN keys(n) WHERE propName ENDS WITH 'embedding' | n[propName]] "
            "    END AS embeddings "
            "WITH n, embeddings, [embedding IN embeddings | gds.similarity.cosine($inputEmbedding, embedding)] AS similarities "
            "WITH n, reduce(s = 0, sim IN similarities | s + sim) AS totalSimilarity "
            "WITH n, totalSimilarity, "
            "    apoc.map.removeKeys(properties(n), [propName IN keys(n) WHERE propName ENDS WITH 'embedding' | propName]) AS filtered_properties "
            "RETURN filtered_properties AS node "
            "ORDER BY totalSimilarity DESC "
            "LIMIT $topK"
        )

        params = {"inputEmbedding": input_embedding, "topK": topK}

        try:
            result = graph_db.query(cypher, params=params)
            if result and len(result) > 0:
                return result
            else:
                return None
        except Exception as e:
            print("GraphRetriever error:", e)
            return None

    # 从实体间查找路径
    def search_paths_from_entities(self, graph_db, from_entity, to_entity) -> dict:
        ### 给start和end的entity，找到路径
        cypher = (
            "MATCH p = allShortestPaths((startNode)-[*..3]-(endNode)) "
            "WHERE startNode.id = $startId AND endNode.id = $endId AND $startId <> $endId "
            "WITH p, "
            "    [n IN nodes(p) | apoc.map.removeKeys(properties(n), "
            "        [propName IN keys(n) WHERE propName ENDS WITH 'embedding' | propName])"
            "    ] AS filtered_nodes, "
            "    [n IN nodes(p) WHERE n.source_text IS NOT NULL | n.source_text] AS sourceTexts "
            "RETURN filtered_nodes AS nodes, sourceTexts"
        )
        params = {"startId": from_entity, "endId": to_entity}

        try:
            result = graph_db.query(cypher, params=params)
            # result = [
            #     {"p": "路径", "sourceTexts": 实体对应的原文},
            #     {"p": "路径", "sourceTexts": 实体对应的原文},
            #     {"p": "路径", "sourceTexts": 实体对应的原文}
            # ]

            # todo: 记得将这里面的p和sourceTexts给区分开    
            if result and len(result) > 0:
                paths = [item["nodes"] for item in result]
                sourceTexts = [item["sourceTexts"] for item in result]
                combined = paths + sourceTexts
                return combined
            else:
                return []
        except Exception as e:
            print("GraphRetriever error:", e)
            return []

