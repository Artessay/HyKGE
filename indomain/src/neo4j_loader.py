# src/neo4j_loader.py
import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain_community.graphs import Neo4jGraph
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents.base import Document
import pandas as pd

class Neo4jLoader:
    """Handles loading of KG data into Neo4j with merge operations and conflict resolution"""
    
    def __init__(self, config: Dict):
        """
        Initialize Neo4j connection with constraints.
        
        Args:
            config: Connection config with url, username, password, database.
        """
        self.graph = Neo4jGraph(
            url=config['url'],
            username=config['username'],
            password=config['password'],
            database=config.get('database', 'neo4j')
        )
        self._setup_constraints()
        self.log_dir = os.path.join("./results", "logs")
        self._init_logger()

    def _init_logger(self):
        """Initialize logging configuration."""
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"loader_{timestamp}.log")

        self.logger = logging.getLogger("Neo4jLoader")
        self.logger.setLevel(logging.INFO)

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def _setup_constraints(self):
        """Create necessary database constraints for data integrity."""
        constraints = [
            "CREATE CONSTRAINT unique_entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT unique_document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
            "CREATE CONSTRAINT unique_chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE"
        ]
        for constraint in constraints:
            try:
                self.graph.query(constraint)
            except Exception as e:
                self.logger.error(f"Constraint setup failed: {str(e)}")
                raise

    def load_all(self, root_dir: str, df: pd.DataFrame):
        """
        Load all documents from the results directory.
        
        Args:
            root_dir: Path to results directory containing document folders.
            df: Documents dataframe for metadata validation.
        """
        self.logger.info("Starting bulk load from %s", root_dir)
        doc_dirs = [d for d in os.listdir(root_dir)
                    if os.path.isdir(os.path.join(root_dir, d))]
        
        for doc_dir in doc_dirs:
            doc_id = doc_dir
            doc_idx = self._get_doc_index(df, doc_id)
            if doc_idx is not None:
                self.load_single_document(doc_idx, root_dir, df)
            else:
                self.logger.warning("Skipping undocumented folder: %s", doc_dir)

    def load_single_document(self, doc_idx: int, root_dir: str, df: pd.DataFrame):
        """
        Load specific document by dataframe index with validation.
        
        Args:
            doc_idx: Index in documents dataframe.
            root_dir: Path to results directory.
            df: Documents dataframe for metadata lookup.
        """
        try:
            doc_id = df.iloc[doc_idx]["Document Number"]
            doc_path = os.path.join(root_dir, doc_id)
            
            if not os.path.exists(doc_path):
                self.logger.error("Document folder not found: %s", doc_id)
                return

            self.logger.info("Processing document: %s (Index: %d)", doc_id, doc_idx)
            kg_data = self._load_document_pages(doc_path)
            self._process_and_load(kg_data, doc_id)
            
        except Exception as e:
            self.logger.error("Failed to load document index %d: %s", doc_idx, str(e))
            raise

    def _load_document_pages(self, doc_path: str) -> List[Dict]:
        """Load and validate all page JSONs for a document."""
        kg_data = []
        page_files = sorted(
            [f for f in os.listdir(doc_path) 
             if f.startswith("page_") and f.endswith(".json") and "_error" not in f],
            key=lambda x: int(x.split('_')[1].split('.')[0])
        )
        
        for page_file in page_files:
            page_path = os.path.join(doc_path, page_file)
            try:
                with open(page_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if self._validate_kg_data(data):
                        kg_data.append(data)
            except Exception as e:
                self.logger.error("Error loading %s: %s", page_file, str(e))
        
        return kg_data

    def _process_and_load(self, kg_data: List[Dict], doc_id: str):
        """Core processing and loading logic."""
        merged_entities = {}
        entity_types = {}
        relationships = []
        
        # Phase 1: Entity merging and source tracking.
        for data in kg_data:
            self._process_entities(data, merged_entities)
    
            for entity in data.get('entities', []):
                entity_id = entity['id']
                entity_types[entity_id] = self._format_label(entity['type'])
            
            relationships.extend(
                self._validate_relationships(
                    data.get('relationships', []),
                    entity_types
                )
            )

        # Phase 2: Node creation with merged properties.
        nodes = [self._create_entity_node(eid, details) for eid, details in merged_entities.items()]
        
        # Phase 3: Relationship creation.
        valid_rels = self._create_relationships(relationships)
        
        # Phase 4: Neo4j transaction.
        try:
            graph_doc = GraphDocument(
                nodes=nodes,
                relationships=valid_rels,
                source=Document(page_content=f"Document {doc_id}")
            )
            self.graph.add_graph_documents([graph_doc])
            self.logger.info("Successfully loaded %d entities and %d relationships for %s",
                             len(nodes), len(valid_rels), doc_id)
        except Exception as e:
            self.logger.error("Transaction failed for %s: %s", doc_id, str(e))
            raise

    def _process_entities(self, data: Dict, merged_entities: Dict):
        """Merge entities across pages with source tracking."""
        for entity in data.get('entities', []):
            entity_id = entity['id']
            properties = {}
            for k, v in entity.get('properties', {}).items():
                if isinstance(v, (dict, list)):
                    properties[k] = json.dumps(v, ensure_ascii=False)
                else:
                    properties[k] = v
            
            if entity_id in merged_entities:
                existing = merged_entities[entity_id]
                existing['properties'] = self._merge_properties(existing['properties'], properties)
            else:
                merged_entities[entity_id] = {
                    'type': self._format_label(entity['type']),
                    'properties': properties
                }

    def _merge_properties(self, existing: Dict, new: Dict) -> Dict:
        """Merge properties with type-aware conflict resolution."""
        merged = existing.copy()
        for key, new_val in new.items():
            if key not in merged:
                merged[key] = new_val
            else:
                if isinstance(new_val, list) and isinstance(merged[key], list):
                    merged[key] = list(set(merged[key] + new_val))
                elif isinstance(new_val, dict) and isinstance(merged[key], dict):
                    merged[key].update(new_val)
                else:
                    # For strings, update only if the new value is longer than the existing one.
                    merged[key] = new_val if isinstance(new_val, str) and len(new_val) > len(str(merged[key])) else merged[key]
        return merged

    def _create_entity_node(self, entity_id: str, details: Dict) -> Node:
        """Create Node object with formatted properties."""
        return Node(
            id=entity_id,
            type=details['type'],
            properties=details['properties']
        )

    def _create_relationships(self, relationships: List[Dict]) -> List[Relationship]:
        """Create validated relationship objects."""
        valid_rels = []
        for rel in relationships:
            try:
                valid_rels.append(Relationship(
                    source=Node(id=rel['source_id'], type=rel['source_type']),
                    target=Node(id=rel['target_id'], type=rel['target_type']),
                    type=self._format_relation_type(rel['type']),
                    properties=rel.get('properties', {})
                ))
            except KeyError as e:
                self.logger.warning("Invalid relationship skipped: %s", str(e))
        return valid_rels

    def _validate_kg_data(self, data: Dict) -> bool:
        """Validate KG data structure."""
        return all(key in data for key in ['entities', 'relationships']) and not data.get('error', False)

    def _validate_relationships(self, relationships: List[Dict], entity_types: Dict) -> List[Dict]:
        """Validate relationship structure."""
        valid = []
        for rel in relationships:
            if all(k in rel for k in ['source_node_id', 'target_node_id', 'type']):
                valid.append({
                    "source_id": rel['source_node_id'],
                    "source_type": entity_types.get(rel['source_node_id'], 'Entity'),
                    "target_id": rel['target_node_id'],
                    "target_type": entity_types.get(rel['target_node_id'], 'Entity'),
                    "type": rel['type'],
                    "properties": rel.get('properties', {})
                })
        return valid

    @staticmethod
    def _get_doc_index(df: pd.DataFrame, doc_id: str) -> Optional[int]:
        """Find dataframe index for a document ID."""
        matches = df[df["Document Number"] == doc_id]
        return matches.index[0] if not matches.empty else None

    @staticmethod
    def _format_key(key: str) -> str:
        """Standardize property keys to camelCase."""
        # parts = key.lower().replace('_', ' ').split()
        # return parts[0] + ''.join(p.capitalize() for p in parts[1:])
        return key

    @staticmethod
    def _format_label(label: str) -> str:
        """Standardize labels to PascalCase (spaces replaced with underscores)."""
        return label.replace(' ', '_')
    
    @staticmethod
    def _format_relation_type(rel_type: str) -> str:
        """Standardize relationship types to UPPER_SNAKE_CASE."""
        return rel_type.upper().replace(' ', '_')

    def resolve_duplicate_labels(self):
        """
        Resolve duplicate nodes with the same id but different labels.
        
        For example, if there are two nodes with id "SOP-012045_S4.8" where one node has 
        the label 'Section' (with fewer properties) and the other 'Subsection' (with more complete
        properties), this function will:
          1. Merge properties from the 'Section' node into the 'Subsection' node (only updating 
             when the property is missing or appears less complete in the 'Subsection' node).
          2. Delete the duplicate 'Section' node.
          3. Add the 'Section' label to the 'Subsection' node so that label information is preserved.
          
        This function should be called after all documents have been loaded.
        """
        try:
            merge_query = """
            // Match Section and Subsection nodes with the same id
            MATCH (sec:Section), (sub:Subsection)
            WHERE sec.id = sub.id
            WITH sec, sub, size(keys(sec)) AS secPropCount, size(keys(sub)) AS subPropCount
            // Proceed only if the Section node has fewer properties than the Subsection node
            WHERE secPropCount < subPropCount
            // Merge properties from the Section node into the Subsection node
            CALL {
              WITH sec, sub
              UNWIND keys(sec) AS key
              WITH sec, sub, key, sec[key] AS secValue, sub[key] AS subValue
              WITH sec, sub, key, secValue, subValue,
                   CASE 
                     WHEN subValue IS NULL OR toString(subValue) = '' THEN true
                     WHEN toString(secValue) <> '' AND size(toString(secValue)) > size(toString(subValue)) THEN true 
                     ELSE false 
                   END AS shouldUpdate
              WHERE shouldUpdate
              // Update the Subsection node's property only if needed
              SET sub[key] = secValue
              RETURN count(*) AS dummy
            }
            // Add the Section label to the Subsection node to preserve label information
            SET sub :Section
            // Delete the duplicate Section node
            DETACH DELETE sec
            """
            self.graph.query(merge_query)
            self.logger.info("Duplicate nodes with duplicate ids resolved successfully.")
        except Exception as e:
            self.logger.error("Failed to resolve duplicate nodes: %s", str(e))
            raise

    def merge_generic_entity_nodes(self):
        """
        Merge generic 'Entity' nodes into their corresponding specialized nodes.
        
        This function finds cases where there exists a generic node (with label 'Entity') and 
        a specialized node (for example, with label 'SRD_Document', 'Person', etc.) that share the same id.
        It uses APOC's mergeNodes procedure to:
          1. Combine the properties of the specialized node and the generic node.
          2. Transfer all relationships from the generic node to the specialized node.
          3. Remove the generic 'Entity' node to avoid duplication.
        
        Ensure that APOC is installed and enabled on your Neo4j instance before running this function.
        """
        try:
            merge_query = """
            MATCH (spec)
            WHERE NOT spec:Entity
            WITH spec
            MATCH (gen:Entity {id: spec.id})
            CALL apoc.refactor.mergeNodes([spec, gen], {properties:"combine", mergeRels:true})
            YIELD node
            RETURN count(node) AS mergedCount
            """
            result = self.graph.query(merge_query)
            self.logger.info("Merged generic 'Entity' nodes into specialized nodes. Merged count: %s", result)
        except Exception as e:
            self.logger.error("Failed to merge generic 'Entity' nodes: %s", str(e))
            raise

#     def merge_all_duplicate_nodes(self):
#         """
#         Merge all nodes with duplicate ids into a single node.
        
#         For each group of nodes sharing the same id, this function will merge their labels,
#         properties, and relationships into one node using APOC's mergeNodes procedure.
        
#         This is useful when there are multiple nodes (e.g. Abbreviation, Organization, Department,
#         Document, etc.) that should represent the same real-world entity.
        
#         Ensure that APOC is installed and enabled on your Neo4j instance before running this function.
#         """
#         try:
#             merge_query = """
#             MATCH (n)
#             WITH n.id AS nodeId, collect(n) AS nodes
#             WHERE size(nodes) > 1
#             CALL apoc.refactor.mergeNodes(nodes, {properties:'combine', mergeRels:true})
#             YIELD node
#             RETURN count(node) AS mergedCount
#             """
#             result = self.graph.query(merge_query)
#             self.logger.info("Merged all duplicate nodes. Merged count: %s", result)
#         except Exception as e:
#             self.logger.error("Failed to merge all duplicate nodes: %s", str(e))
#             raise


    def create_parent_relationships(self):
        try:
            # Process Section nodes
            section_query = """
            MATCH (s:Section)
            WHERE s.id CONTAINS '_'
            WITH s, split(s.id, '_')[0] AS doc_id
            MATCH (d {id: doc_id})
            MERGE (d)-[:CONTAINS]->(s)
            """
            result = self.graph.query(section_query)
            self.logger.info("Created parent relationships for orphan nodes. Count: %s", result)

            # Process Document_Metadata nodes
            metadata_query = """
            MATCH (m:Document_Metadata)
            WHERE m.id CONTAINS '_'
            WITH m, split(m.id, '_')[0] AS doc_id
            MATCH (d {id: doc_id})
            MERGE (d)-[:CONTAINS]->(m)
            """
            result = self.graph.query(metadata_query)
            self.logger.info("Processed Document metadata nodes. Count: %s", result)
        except Exception as e:
            self.logger.error("Failed to create parent relationships: %s", str(e))
            raise

    def merge_misidentified_documents(self):
        """
        Merge documents that appear to be duplicates due to ID extraction errors.
        
        This function finds document nodes that have similar attributes (title, language, status, 
        status_date, and version) but different IDs. It merges them using APOC's mergeNodes
        procedure, ensuring that the correct properties and relationships are preserved.
        
        Ensure that APOC is installed and enabled on your Neo4j instance before running this function.
        """
        try:
            merge_query = """
            MATCH (d1)
            MATCH (d2)
            WHERE d1.id <> d2.id
              AND d1.title = d2.title
              AND d1.language = d2.language
              AND d1.status = d2.status
              AND d1.status_date = d2.status_date
              AND d1.version = d2.version
            WITH collect(d1) + collect(d2) AS nodes
            CALL apoc.refactor.mergeNodes(nodes, {properties:'combine', mergeRels:true})
            YIELD node
            RETURN count(node) AS mergedCount
            """
            result = self.graph.query(merge_query)
            self.logger.info("Merged misidentified document nodes. Merged count: %s", result)
        except Exception as e:
            self.logger.error("Failed to merge misidentified documents: %s", str(e))
            raise
