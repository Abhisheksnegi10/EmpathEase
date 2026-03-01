"""
Semantic Memory - Neo4j Knowledge Graph Integration.

Maintains knowledge graph of user's:
- Relationships (family, friends, colleagues)
- Entities (people, places, organizations)
- Goals and concerns
- Long-term patterns

Supports context queries for personalized responses.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Represents a known entity in user's world."""
    name: str
    entity_type: str  # person, place, organization, goal, concern
    relationship: Optional[str] = None  # How it relates to user
    sentiment: float = 0.0  # User's feeling toward it (-1 to 1)
    last_mentioned: Optional[datetime] = None
    mention_count: int = 0
    properties: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


@dataclass
class Relationship:
    """Represents a relationship between entities."""
    from_entity: str
    to_entity: str
    relationship_type: str
    strength: float = 0.5  # 0 to 1
    sentiment: float = 0.0  # -1 to 1
    last_updated: Optional[datetime] = None


class SemanticMemoryManager:
    """
    Manages semantic memory via Neo4j knowledge graph.
    
    Features:
    - Entity extraction and storage
    - Relationship tracking
    - Context queries for LLM
    - Pattern detection
    """
    
    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "empathease123",
    ):
        """
        Initialize semantic memory manager.
        
        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Database username
            neo4j_password: Database password
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self._driver = None
        self._initialized = False
        
        logger.info(f"SemanticMemoryManager created (neo4j: {neo4j_uri})")
    
    async def initialize(self) -> bool:
        """
        Initialize Neo4j connection.
        
        Returns:
            True if successful.
        """
        if self._initialized:
            return True
        
        try:
            from neo4j import GraphDatabase
            
            self._driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password)
            )
            
            # Verify connection
            with self._driver.session() as session:
                session.run("RETURN 1")
            
            # Create indexes
            await self._create_indexes()
            
            self._initialized = True
            logger.info("Semantic memory initialized successfully")
            return True
            
        except ImportError:
            logger.error("neo4j driver not installed. Run: pip install neo4j")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize semantic memory: {e}")
            return False
    
    async def _create_indexes(self) -> None:
        """Create database indexes."""
        with self._driver.session() as session:
            # Index on user_id for all node types
            session.run("""
                CREATE INDEX IF NOT EXISTS FOR (u:User) ON (u.user_id)
            """)
            session.run("""
                CREATE INDEX IF NOT EXISTS FOR (p:Person) ON (p.name)
            """)
            session.run("""
                CREATE INDEX IF NOT EXISTS FOR (o:Organization) ON (o.name)
            """)
    
    def _hash_user_id(self, user_id: str) -> str:
        """Hash user ID for privacy."""
        return hashlib.sha256(user_id.encode()).hexdigest()[:16]
    
    async def add_entity(
        self,
        user_id: str,
        entity_name: str,
        entity_type: str,
        relationship: Optional[str] = None,
        sentiment: float = 0.0,
        properties: Optional[Dict] = None,
    ) -> bool:
        """
        Add or update an entity in user's knowledge graph.
        
        Args:
            user_id: User identifier
            entity_name: Name of the entity
            entity_type: Type (person, place, organization, goal, concern)
            relationship: How it relates to user (e.g., "sister", "employer")
            sentiment: User's feeling toward entity (-1 to 1)
            properties: Additional properties
        
        Returns:
            True if successful.
        """
        if not await self.initialize():
            return False
        
        try:
            user_hash = self._hash_user_id(user_id)
            props = properties or {}
            
            with self._driver.session() as session:
                # Create/merge user node
                session.run("""
                    MERGE (u:User {user_id: $user_id})
                """, user_id=user_hash)
                
                # Create/update entity and relationship
                label = entity_type.capitalize()
                query = f"""
                    MATCH (u:User {{user_id: $user_id}})
                    MERGE (e:{label} {{name: $name}})
                    ON CREATE SET 
                        e.first_mentioned = datetime(),
                        e.mention_count = 1
                    ON MATCH SET 
                        e.mention_count = e.mention_count + 1,
                        e.last_mentioned = datetime()
                    MERGE (u)-[r:KNOWS]->(e)
                    SET r.relationship = $relationship,
                        r.sentiment = $sentiment,
                        r.last_updated = datetime()
                """
                
                session.run(
                    query,
                    user_id=user_hash,
                    name=entity_name,
                    relationship=relationship,
                    sentiment=sentiment,
                )
                
                # Add properties
                if props:
                    for key, value in props.items():
                        session.run(f"""
                            MATCH (e:{label} {{name: $name}})
                            SET e.{key} = $value
                        """, name=entity_name, value=value)
            
            logger.info(f"Added entity: {entity_name} ({entity_type}) for user {user_hash}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add entity: {e}")
            return False
    
    async def get_user_context(
        self,
        user_id: str,
        limit: int = 10,
    ) -> Dict[str, List[Entity]]:
        """
        Get user's full context (all known entities and relationships).
        
        Args:
            user_id: User identifier
            limit: Max entities per type
        
        Returns:
            Dict mapping entity types to entity lists.
        """
        if not await self.initialize():
            return {}
        
        try:
            user_hash = self._hash_user_id(user_id)
            context = {}
            
            with self._driver.session() as session:
                result = session.run("""
                    MATCH (u:User {user_id: $user_id})-[r:KNOWS]->(e)
                    RETURN labels(e)[0] as type, e.name as name, 
                           r.relationship as rel, r.sentiment as sentiment,
                           e.mention_count as mentions
                    ORDER BY e.mention_count DESC
                    LIMIT $limit
                """, user_id=user_hash, limit=limit * 5)
                
                for record in result:
                    entity_type = record["type"].lower()
                    entity = Entity(
                        name=record["name"],
                        entity_type=entity_type,
                        relationship=record["rel"],
                        sentiment=record["sentiment"] or 0.0,
                        mention_count=record["mentions"] or 0,
                    )
                    
                    if entity_type not in context:
                        context[entity_type] = []
                    if len(context[entity_type]) < limit:
                        context[entity_type].append(entity)
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to get user context: {e}")
            return {}
    
    async def query_relationship(
        self,
        user_id: str,
        entity_name: str,
    ) -> Optional[Entity]:
        """
        Query specific relationship with an entity.
        
        Args:
            user_id: User identifier
            entity_name: Name of the entity
        
        Returns:
            Entity object if found.
        """
        if not await self.initialize():
            return None
        
        try:
            user_hash = self._hash_user_id(user_id)
            
            with self._driver.session() as session:
                result = session.run("""
                    MATCH (u:User {user_id: $user_id})-[r:KNOWS]->(e)
                    WHERE toLower(e.name) CONTAINS toLower($name)
                    RETURN labels(e)[0] as type, e.name as name,
                           r.relationship as rel, r.sentiment as sentiment,
                           e.mention_count as mentions, e.last_mentioned as last
                    LIMIT 1
                """, user_id=user_hash, name=entity_name)
                
                record = result.single()
                if record:
                    return Entity(
                        name=record["name"],
                        entity_type=record["type"].lower(),
                        relationship=record["rel"],
                        sentiment=record["sentiment"] or 0.0,
                        mention_count=record["mentions"] or 0,
                        last_mentioned=record["last"],
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to query relationship: {e}")
            return None
    
    async def extract_and_store_entities(
        self,
        user_id: str,
        text: str,
    ) -> List[str]:
        """
        Extract entities from text and store in graph.
        
        Uses simple NER patterns. In production, use spaCy.
        
        Args:
            user_id: User identifier
            text: Text to extract from
        
        Returns:
            List of extracted entity names.
        """
        import re
        
        extracted = []
        
        # Simple extraction patterns
        patterns = [
            (r'my (\w+) (\w+)', 'person'),  # "my sister Sarah"
            (r'(\w+) is my (\w+)', 'person'),  # "Sarah is my sister"
            (r'at (\w+)', 'organization'),  # "at Google"
            (r'work at (\w+)', 'organization'),  # "work at TechCorp"
        ]
        
        # Capitalize words that might be names
        words = text.split()
        for i, word in enumerate(words):
            if i > 0 and word[0].isupper() and len(word) > 1:
                clean = re.sub(r'[^\w]', '', word)
                if clean and clean not in ['I', "I'm", "I've", "I'll"]:
                    await self.add_entity(
                        user_id=user_id,
                        entity_name=clean,
                        entity_type='person',
                    )
                    extracted.append(clean)
        
        return extracted
    
    async def get_context_for_llm(
        self,
        user_id: str,
    ) -> str:
        """
        Get formatted context for LLM prompt injection.
        
        Args:
            user_id: User identifier
        
        Returns:
            Formatted context string.
        """
        context = await self.get_user_context(user_id, limit=5)
        
        if not context:
            return ""
        
        lines = ["[USER CONTEXT: Known relationships and topics]"]
        
        for entity_type, entities in context.items():
            if entities:
                entity_strs = [
                    f"{e.name} ({e.relationship})" if e.relationship else e.name
                    for e in entities[:3]
                ]
                lines.append(f"- {entity_type}s: {', '.join(entity_strs)}")
        
        return "\n".join(lines)
    
    async def delete_user_data(self, user_id: str) -> bool:
        """
        Delete all graph data for a user (GDPR compliance).
        
        Args:
            user_id: User identifier
        
        Returns:
            True if successful.
        """
        if not await self.initialize():
            return False
        
        try:
            user_hash = self._hash_user_id(user_id)
            
            with self._driver.session() as session:
                # Delete user node and all relationships
                session.run("""
                    MATCH (u:User {user_id: $user_id})
                    DETACH DELETE u
                """, user_id=user_hash)
            
            logger.info(f"Deleted all graph data for user: {user_hash}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete user data: {e}")
            return False
    
    def close(self):
        """Close database connection."""
        if self._driver:
            self._driver.close()
            self._initialized = False


# Singleton
_manager: Optional[SemanticMemoryManager] = None


def get_semantic_memory() -> SemanticMemoryManager:
    """Get singleton semantic memory manager."""
    global _manager
    if _manager is None:
        _manager = SemanticMemoryManager()
    return _manager
