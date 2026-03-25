"""Entity extraction from conversations."""

from typing import Any

from nanobot.agent.memory_v2.models import Entity, Relationship


class EntityExtractor:
    """
    Extract entities and relationships from text.

    Uses spaCy for named entity recognition.
    """

    def __init__(
        self, enabled: bool = True, min_confidence: float = 0.7, types: list[str] | None = None
    ) -> None:
        """
        Initialize entity extractor.

        Args:
            enabled: Whether entity extraction is enabled
            min_confidence: Minimum confidence threshold
            types: Entity types to extract (PERSON, ORG, etc.)
        """
        self.enabled = enabled
        self.min_confidence = min_confidence
        self.types = types or ["PERSON", "ORG", "GPE", "PRODUCT", "EVENT"]

        self._nlp = None
        if enabled:
            self._init_nlp()

    def _init_nlp(self) -> None:
        """Initialize spaCy NLP model."""
        try:
            import spacy

            # Try to load English model
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                # Model not installed, try to download
                import subprocess
                import sys

                print("Downloading spaCy model en_core_web_sm...")
                subprocess.check_call(
                    [sys.executable, "-m", "spacy", "download", "en_core_web_sm"]
                )
                self._nlp = spacy.load("en_core_web_sm")
        except ImportError:
            print(
                "Warning: spaCy not installed. Entity extraction disabled. "
                "Install with: pip install spacy"
            )
            self.enabled = False

    def extract(self, text: str) -> list[Entity]:
        """
        Extract entities from text.

        Args:
            text: Text to extract entities from

        Returns:
            List of extracted entities
        """
        if not self.enabled or self._nlp is None:
            return []

        doc = self._nlp(text)
        entities = []

        for ent in doc.ents:
            if ent.label_ in self.types:
                entity = Entity(
                    name=ent.text,
                    type=self._normalize_type(ent.label_),
                    confidence=1.0,  # spaCy doesn't provide confidence
                    attributes={
                        "spacy_label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char,
                    },
                )
                entities.append(entity)

        return entities

    def extract_relationships(self, text: str, entities: list[Entity]) -> list[Relationship]:
        """
        Extract relationships between entities.

        This is a simple implementation that looks for
        co-occurrence and common patterns.

        Args:
            text: Text to analyze
            entities: List of entities in the text

        Returns:
            List of relationships
        """
        if not self.enabled or len(entities) < 2:
            return []

        relationships = []
        entity_names = {e.name for e in entities}

        # Simple co-occurrence relationships
        for i, e1 in enumerate(entities):
            for e2 in entities[i + 1 :]:
                # Check if entities appear close together
                if self._are_entities_close(text, e1, e2, max_distance=50):
                    rel_type = self._infer_relationship(text, e1, e2)
                    if rel_type:
                        relationships.append(
                            Relationship(source=e1.name, target=e2.name, type=rel_type)
                        )

        return relationships

    def _normalize_type(self, spacy_type: str) -> str:
        """Normalize spaCy entity type to our schema."""
        mapping = {
            "PERSON": "person",
            "ORG": "organization",
            "GPE": "location",  # Geopolitical entity
            "PRODUCT": "product",
            "EVENT": "event",
            "WORK_OF_ART": "work",
            "LAW": "law",
            "LANGUAGE": "language",
            "DATE": "date",
            "TIME": "time",
            "PERCENT": "percent",
            "MONEY": "money",
            "QUANTITY": "quantity",
            "CARDINAL": "number",
            "ORDINAL": "ordinal",
        }
        return mapping.get(spacy_type, spacy_type.lower())

    def _are_entities_close(
        self, text: str, e1: Entity, e2: Entity, max_distance: int = 50
    ) -> bool:
        """Check if two entities appear close together in text."""
        pos1 = text.find(e1.name)
        pos2 = text.find(e2.name)

        if pos1 == -1 or pos2 == -1:
            return False

        return abs(pos1 - pos2) <= max_distance

    def _infer_relationship(self, text: str, e1: Entity, e2: Entity) -> str | None:
        """
        Infer relationship type between two entities.

        This is a simple rule-based approach.
        """
        text_lower = text.lower()

        # Check for relationship indicators
        if "works for" in text_lower or "employed by" in text_lower:
            if e1.type == "person" and e2.type == "organization":
                return "works_for"
        elif "knows" in text_lower or "met" in text_lower:
            if e1.type == "person" and e2.type == "person":
                return "knows"
        elif "located in" in text_lower or "based in" in text_lower:
            if e2.type == "location":
                return "located_in"
        elif "created" in text_lower or "developed" in text_lower:
            return "created"

        return None
