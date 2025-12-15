"""
Contextual Memory & Experience Retrieval System (RAG-like Pattern Matching)

NOTE ON TERMINOLOGY: This module is named "recursive_learning" but it does NOT
perform gradient-based learning (SGD) or update model weights. The LLM itself
does not "learn" or improve from this system.

WHAT THIS ACTUALLY DOES:
- Stores successful task inputs/outputs in a persistent knowledge store
- Retrieves similar past experiences via context hashing when new tasks arrive
- Tracks patterns of successful actions for different contexts
- Builds "skills" by aggregating patterns (metadata, not neural updates)

This is effectively a Retrieval-Augmented Generation (RAG) pattern for agent
memory. It creates a persistent "memory" of what worked, enabling the agent to
suggest actions based on past successes WITHOUT modifying the underlying LLM.

The term "learning" here refers to accumulating experience data, not machine
learning in the SGD sense.
"""

import asyncio
import logging
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar
from enum import Enum
from datetime import datetime, timedelta
from collections import deque
import hashlib
import statistics
import pickle
import pickle
from pathlib import Path

try:
    import chromadb
    from sentence_transformers import SentenceTransformer

    HAS_RAG = True
except ImportError:
    HAS_RAG = False

logger = logging.getLogger(__name__)


class LearningSignal(Enum):
    """Types of learning signals"""

    POSITIVE = "positive"  # Successful outcome
    NEGATIVE = "negative"  # Failed outcome
    NEUTRAL = "neutral"  # Uncertain outcome
    CORRECTIVE = "corrective"  # External correction


class ExperienceType(Enum):
    """Types of experiences to learn from"""

    TASK_COMPLETION = "task_completion"
    ERROR_RECOVERY = "error_recovery"
    OPTIMIZATION = "optimization"
    COLLABORATION = "collaboration"
    USER_FEEDBACK = "user_feedback"


@dataclass
class Experience:
    """A learning experience"""

    experience_id: str
    experience_type: ExperienceType
    context: Dict[str, Any]
    action: str
    outcome: Dict[str, Any]
    signal: LearningSignal
    reward: float  # -1.0 to 1.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experienceId": self.experience_id,
            "experienceType": self.experience_type.value,
            "context": self.context,
            "action": self.action,
            "outcome": self.outcome,
            "signal": self.signal.value,
            "reward": self.reward,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class Pattern:
    """A learned pattern"""

    pattern_id: str
    pattern_type: str
    conditions: List[Dict[str, Any]]
    action: str
    expected_outcome: str
    confidence: float
    usage_count: int = 0
    success_count: int = 0
    last_used: Optional[str] = None

    @property
    def success_rate(self) -> float:
        if self.usage_count == 0:
            return 0.0
        return self.success_count / self.usage_count


@dataclass
class Skill:
    """A learned skill"""

    skill_id: str
    name: str
    description: str
    patterns: List[str]  # Pattern IDs
    proficiency: float  # 0.0 to 1.0
    experience_count: int = 0

    def update_proficiency(self, delta: float):
        """Update proficiency with bounds"""
        self.proficiency = max(0.0, min(1.0, self.proficiency + delta))


class ExperienceBuffer:
    """Replay buffer for experiences"""

    def __init__(self, max_size: int = 10000, priority_sampling: bool = True):
        self.max_size = max_size
        self.priority_sampling = priority_sampling
        self._buffer: deque = deque(maxlen=max_size)
        self._priorities: Dict[str, float] = {}

    def add(self, experience: Experience, priority: float = 1.0):
        """Add experience to buffer"""
        self._buffer.append(experience)
        self._priorities[experience.experience_id] = priority

        # Cleanup old priorities
        if len(self._priorities) > self.max_size * 1.5:
            current_ids = {e.experience_id for e in self._buffer}
            self._priorities = {
                k: v for k, v in self._priorities.items() if k in current_ids
            }

    def sample(self, batch_size: int) -> List[Experience]:
        """Sample experiences from buffer"""
        if len(self._buffer) <= batch_size:
            return list(self._buffer)

        if self.priority_sampling:
            # Priority-based sampling
            import random

            weights = [self._priorities.get(e.experience_id, 1.0) for e in self._buffer]
            total = sum(weights)
            weights = [w / total for w in weights]

            indices = random.choices(
                range(len(self._buffer)), weights=weights, k=batch_size
            )
            return [self._buffer[i] for i in indices]
        else:
            import random

            return random.sample(list(self._buffer), batch_size)

    def get_recent(self, count: int) -> List[Experience]:
        """Get most recent experiences"""
        return list(self._buffer)[-count:]

    def get_by_type(self, exp_type: ExperienceType) -> List[Experience]:
        """Get experiences by type"""
        return [e for e in self._buffer if e.experience_type == exp_type]

    def __len__(self) -> int:
        return len(self._buffer)


class PatternExtractor:
    """Extracts patterns from experiences"""

    def __init__(self, min_occurrences: int = 3, min_confidence: float = 0.7):
        self.min_occurrences = min_occurrences
        self.min_confidence = min_confidence
        self._pattern_candidates: Dict[str, Dict[str, Any]] = {}

    def analyze(self, experiences: List[Experience]) -> List[Pattern]:
        """Extract patterns from experiences"""
        # Group by context similarity
        context_groups = self._group_by_context(experiences)

        patterns = []
        for group_key, group in context_groups.items():
            if len(group) >= self.min_occurrences:
                pattern = self._extract_pattern(group)
                if pattern and pattern.confidence >= self.min_confidence:
                    patterns.append(pattern)

        return patterns

    def _group_by_context(
        self, experiences: List[Experience]
    ) -> Dict[str, List[Experience]]:
        """Group experiences by similar context"""
        groups = {}

        for exp in experiences:
            # Create context signature
            context_keys = sorted(exp.context.keys())
            signature = hashlib.md5(
                json.dumps(context_keys, sort_keys=True).encode()
            ).hexdigest()[:8]

            if signature not in groups:
                groups[signature] = []
            groups[signature].append(exp)

        return groups

    def _extract_pattern(self, experiences: List[Experience]) -> Optional[Pattern]:
        """Extract pattern from similar experiences"""
        if not experiences:
            return None

        # Find common conditions
        conditions = self._find_common_conditions(experiences)

        # Find most successful action
        action_success = {}
        for exp in experiences:
            action = exp.action
            if action not in action_success:
                action_success[action] = {"total": 0, "success": 0}
            action_success[action]["total"] += 1
            if exp.signal == LearningSignal.POSITIVE:
                action_success[action]["success"] += 1

        best_action = max(
            action_success.keys(),
            key=lambda a: action_success[a]["success"]
            / max(action_success[a]["total"], 1),
        )

        # Calculate confidence
        stats = action_success[best_action]
        confidence = stats["success"] / stats["total"] if stats["total"] > 0 else 0

        # Find expected outcome
        positive_outcomes = [
            e.outcome
            for e in experiences
            if e.action == best_action and e.signal == LearningSignal.POSITIVE
        ]
        expected_outcome = str(positive_outcomes[0]) if positive_outcomes else "success"

        return Pattern(
            pattern_id=hashlib.md5(f"{conditions}{best_action}".encode()).hexdigest()[
                :12
            ],
            pattern_type="extracted",
            conditions=conditions,
            action=best_action,
            expected_outcome=expected_outcome,
            confidence=confidence,
        )

    def _find_common_conditions(
        self, experiences: List[Experience]
    ) -> List[Dict[str, Any]]:
        """Find conditions common to all experiences"""
        if not experiences:
            return []

        # Start with first experience's context
        common = set(experiences[0].context.keys())

        # Intersect with others
        for exp in experiences[1:]:
            common &= set(exp.context.keys())

        return [{"key": k, "present": True} for k in common]


class SkillBuilder:
    """Builds and manages skills from patterns"""

    def __init__(self):
        self._skills: Dict[str, Skill] = {}
        self._pattern_to_skill: Dict[str, str] = {}

    def create_skill(
        self, name: str, description: str, patterns: List[Pattern]
    ) -> Skill:
        """Create a new skill from patterns"""
        skill_id = hashlib.md5(
            f"{name}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        skill = Skill(
            skill_id=skill_id,
            name=name,
            description=description,
            patterns=[p.pattern_id for p in patterns],
            proficiency=self._calculate_initial_proficiency(patterns),
        )

        self._skills[skill_id] = skill
        for pattern in patterns:
            self._pattern_to_skill[pattern.pattern_id] = skill_id

        return skill

    def _calculate_initial_proficiency(self, patterns: List[Pattern]) -> float:
        """Calculate initial skill proficiency"""
        if not patterns:
            return 0.0

        # Average pattern confidence
        avg_confidence = sum(p.confidence for p in patterns) / len(patterns)

        # Weight by usage
        total_usage = sum(p.usage_count for p in patterns)
        if total_usage > 0:
            weighted_success = (
                sum(p.success_rate * p.usage_count for p in patterns) / total_usage
            )
            return (avg_confidence + weighted_success) / 2

        return avg_confidence * 0.5

    def update_skill(self, skill_id: str, experience: Experience):
        """Update skill based on experience"""
        if skill_id not in self._skills:
            return

        skill = self._skills[skill_id]
        skill.experience_count += 1

        # Update proficiency based on outcome
        if experience.signal == LearningSignal.POSITIVE:
            skill.update_proficiency(0.01)  # Small positive update
        elif experience.signal == LearningSignal.NEGATIVE:
            skill.update_proficiency(-0.02)  # Larger negative update

    def get_skill(self, skill_id: str) -> Optional[Skill]:
        return self._skills.get(skill_id)

    def get_all_skills(self) -> List[Skill]:
        return list(self._skills.values())

    def find_applicable_skills(self, context: Dict[str, Any]) -> List[Skill]:
        """Find skills applicable to context"""
        # Simple matching based on context keys
        applicable = []
        for skill in self._skills.values():
            if skill.proficiency > 0.3:  # Only consider skills with some proficiency
                applicable.append(skill)
        return sorted(applicable, key=lambda s: s.proficiency, reverse=True)


class LearningPolicy:
    """Policy for learning decisions"""

    def __init__(
        self,
        learning_rate: float = 0.01,
        exploration_rate: float = 0.1,
        decay_rate: float = 0.999,
    ):
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.decay_rate = decay_rate
        self._step_count = 0

    def should_explore(self) -> bool:
        """Decide whether to explore or exploit"""
        import random

        current_rate = self.exploration_rate * (self.decay_rate**self._step_count)
        return random.random() < current_rate

    def calculate_update(self, reward: float, confidence: float) -> float:
        """Calculate learning update magnitude"""
        self._step_count += 1
        return self.learning_rate * reward * (1 - confidence)

    def reset(self):
        """Reset policy state"""
        self._step_count = 0


class RecursiveLearner:
    """Main recursive learning system"""

    def __init__(self, storage_path: Optional[Path] = None, buffer_size: int = 10000):
        self.storage_path = storage_path or Path.home() / ".astro" / "learning"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Use SQLite for scalable persistence instead of JSON
        from src.core.database import DatabaseManager

        self._db = DatabaseManager()

        self.experience_buffer = ExperienceBuffer(max_size=buffer_size)
        self.pattern_extractor = PatternExtractor()
        self.skill_builder = SkillBuilder()
        self.policy = LearningPolicy()

        self._patterns: Dict[str, Pattern] = {}
        self._learning_iterations = 0
        self._improvement_history: List[Dict[str, Any]] = []

        # Load existing knowledge
        self._load_knowledge()

        # Initialize RAG system (ChromaDB + Embeddings)
        self.chroma_client = None
        self.collection = None
        self.embedding_model = None
        self._rag_available = False  # Instance-level RAG availability flag

        if HAS_RAG:
            try:
                logger.info(
                    "Initializing Semantic Memory (ChromaDB + SentenceTransformers)..."
                )
                self.chroma_client = chromadb.PersistentClient(
                    path=str(self.storage_path / "chroma")
                )
                self.collection = self.chroma_client.get_or_create_collection(
                    name="experiences"
                )
                # Use a lightweight model for speed
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                self._rag_available = True
                logger.info("Semantic Memory initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Semantic Memory: {e}")
                self._rag_available = False
        else:
            logger.warning(
                "ChromaDB or SentenceTransformers not found. Semantic Memory disabled."
            )

    def record_experience(
        self,
        experience_type: ExperienceType,
        context: Dict[str, Any],
        action: str,
        outcome: Dict[str, Any],
        reward: float,
    ) -> str:
        """Record a new experience"""
        # Determine signal from reward
        if reward > 0.3:
            signal = LearningSignal.POSITIVE
        elif reward < -0.3:
            signal = LearningSignal.NEGATIVE
        else:
            signal = LearningSignal.NEUTRAL

        experience = Experience(
            experience_id=hashlib.md5(
                f"{context}{action}{datetime.now().isoformat()}".encode()
            ).hexdigest()[:12],
            experience_type=experience_type,
            context=context,
            action=action,
            outcome=outcome,
            signal=signal,
            reward=reward,
        )

        # Add to buffer with priority based on reward magnitude
        priority = abs(reward) + 0.5
        self.experience_buffer.add(experience, priority)

        # Trigger incremental learning
        self._incremental_learn(experience)

        # Store in Vector DB for Semantic Retrieval
        if self._rag_available and self.collection and self.embedding_model:
            try:
                # Create embedding for context (stringify context)
                context_str = json.dumps(context, sort_keys=True)
                embedding = self.embedding_model.encode(context_str).tolist()

                self.collection.add(
                    documents=[context_str],
                    embeddings=[embedding],
                    metadatas=[
                        {
                            "action": action,
                            "reward": reward,
                            "signal": signal.value,
                            "type": experience_type.value,
                        }
                    ],
                    ids=[experience.experience_id],
                )
            except Exception as e:
                logger.error(f"Failed to store experience in Vector DB: {e}")

        logger.debug(f"Recorded experience: {experience.experience_id}")
        return experience.experience_id

    def _incremental_learn(self, experience: Experience):
        """Learn incrementally from new experience"""
        # Update relevant patterns
        for pattern in self._patterns.values():
            if self._matches_pattern(experience, pattern):
                pattern.usage_count += 1
                pattern.last_used = datetime.now().isoformat()

                if experience.signal == LearningSignal.POSITIVE:
                    pattern.success_count += 1
                    pattern.confidence = min(1.0, pattern.confidence + 0.01)
                elif experience.signal == LearningSignal.NEGATIVE:
                    pattern.confidence = max(0.0, pattern.confidence - 0.02)

        # Update skills
        for skill_id in self.skill_builder._skills:
            skill = self.skill_builder._skills[skill_id]
            if any(p in skill.patterns for p in self._patterns):
                self.skill_builder.update_skill(skill_id, experience)

    def _matches_pattern(self, experience: Experience, pattern: Pattern) -> bool:
        """Check if experience matches pattern"""
        if experience.action != pattern.action:
            return False

        # Check conditions
        for condition in pattern.conditions:
            key = condition.get("key")
            if key and key not in experience.context:
                return False

        return True

    async def learn_batch(self, batch_size: int = 32) -> Dict[str, Any]:
        """Perform batch learning from experience buffer"""
        self._learning_iterations += 1

        # Sample experiences
        experiences = self.experience_buffer.sample(batch_size)

        # Extract new patterns
        new_patterns = self.pattern_extractor.analyze(experiences)

        # Merge with existing patterns
        patterns_added = 0
        for pattern in new_patterns:
            if pattern.pattern_id not in self._patterns:
                self._patterns[pattern.pattern_id] = pattern
                patterns_added += 1
            else:
                # Update existing pattern
                existing = self._patterns[pattern.pattern_id]
                existing.confidence = (existing.confidence + pattern.confidence) / 2

        # Build/update skills
        if new_patterns:
            # Group patterns by action type
            by_action = {}
            for p in new_patterns:
                if p.action not in by_action:
                    by_action[p.action] = []
                by_action[p.action].append(p)

            for action, patterns in by_action.items():
                if len(patterns) >= 2:
                    skill = self.skill_builder.create_skill(
                        name=f"skill_{action}",
                        description=f"Learned skill for {action}",
                        patterns=patterns,
                    )

        # Calculate improvement metrics
        improvement = self._calculate_improvement()

        self._improvement_history.append(
            {
                "iteration": self._learning_iterations,
                "patterns_added": patterns_added,
                "total_patterns": len(self._patterns),
                "improvement": improvement,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Save knowledge periodically
        if self._learning_iterations % 10 == 0:
            self._save_knowledge()

        return {
            "iteration": self._learning_iterations,
            "experiences_processed": len(experiences),
            "patterns_extracted": len(new_patterns),
            "patterns_added": patterns_added,
            "total_patterns": len(self._patterns),
            "total_skills": len(self.skill_builder._skills),
            "improvement": improvement,
        }

    def _calculate_improvement(self) -> float:
        """Calculate learning improvement"""
        if len(self._improvement_history) < 2:
            return 0.0

        # Compare pattern success rates
        if not self._patterns:
            return 0.0

        avg_success = sum(p.success_rate for p in self._patterns.values()) / len(
            self._patterns
        )
        return avg_success

    def suggest_action(self, context: Dict[str, Any]) -> Optional[Tuple[str, float]]:
        """Suggest best action based on learned patterns"""
        best_action = None
        best_score = 0.0

        # Check if we should explore
        if self.policy.should_explore():
            # Return random pattern's action
            if self._patterns:
                import random

                pattern = random.choice(list(self._patterns.values()))
                return pattern.action, pattern.confidence * 0.5

        # Find best matching pattern
        for pattern in self._patterns.values():
            match_score = self._calculate_match_score(context, pattern)
            if match_score > best_score:
                best_score = match_score
                best_action = pattern.action

        if best_action:
            return best_action, best_score

        # Fallback: Semantic Search (RAG) if no exact pattern match
        if self._rag_available and self.collection and self.embedding_model:
            try:
                context_str = json.dumps(context, sort_keys=True)
                embedding = self.embedding_model.encode(context_str).tolist()

                results = self.collection.query(
                    query_embeddings=[embedding], n_results=5
                )

                if results["ids"] and results["ids"][0]:
                    # Analyze retrieved experiences
                    actions = {}
                    for i, meta in enumerate(results["metadatas"][0]):
                        action = meta["action"]
                        reward = float(meta["reward"])
                        dist = (
                            results["distances"][0][i]
                            if "distances" in results
                            else 0.5
                        )

                        # Score = reward weighted by similarity (1 - dist)
                        score = reward * (1 - min(dist, 1.0))

                        if action not in actions:
                            actions[action] = 0.0
                        actions[action] += score

                    # Pick best action from semantic retrieval
                    if actions:
                        rag_best_action = max(actions.items(), key=lambda x: x[1])
                        if rag_best_action[1] > 0.2:  # Threshold
                            return rag_best_action[0], min(rag_best_action[1], 0.9)
            except Exception as e:
                logger.error(f"Semantic retrieval failed: {e}")

        return None

    def _calculate_match_score(
        self, context: Dict[str, Any], pattern: Pattern
    ) -> float:
        """Calculate how well context matches pattern"""
        if not pattern.conditions:
            return pattern.confidence * 0.5

        matches = 0
        for condition in pattern.conditions:
            key = condition.get("key")
            if key and key in context:
                matches += 1

        match_ratio = matches / len(pattern.conditions) if pattern.conditions else 0
        return match_ratio * pattern.confidence * pattern.success_rate

    def get_knowledge_summary(self) -> Dict[str, Any]:
        """Get summary of learned knowledge"""
        return {
            "total_experiences": len(self.experience_buffer),
            "total_patterns": len(self._patterns),
            "total_skills": len(self.skill_builder._skills),
            "learning_iterations": self._learning_iterations,
            "avg_pattern_confidence": (
                sum(p.confidence for p in self._patterns.values()) / len(self._patterns)
                if self._patterns
                else 0.0
            ),
            "avg_skill_proficiency": (
                sum(s.proficiency for s in self.skill_builder._skills.values())
                / len(self.skill_builder._skills)
                if self.skill_builder._skills
                else 0.0
            ),
            "top_patterns": [
                {
                    "action": p.action,
                    "confidence": p.confidence,
                    "success_rate": p.success_rate,
                }
                for p in sorted(
                    self._patterns.values(), key=lambda x: x.confidence, reverse=True
                )[:5]
            ],
        }

    def _save_knowledge(self):
        """Persist learned knowledge to SQLite database (scalable)"""
        try:
            # Save each pattern to database
            for pattern_id, pattern in self._patterns.items():
                self._db.save_learning_pattern(
                    pattern_id=pattern.pattern_id,
                    pattern_type=pattern.pattern_type,
                    conditions=pattern.conditions,
                    action=pattern.action,
                    expected_outcome=pattern.expected_outcome,
                    confidence=pattern.confidence,
                    usage_count=pattern.usage_count,
                    success_count=pattern.success_count,
                    last_used=pattern.last_used,
                )

            # Save metadata
            self._db.save_learning_metadata(
                "learning_iterations", self._learning_iterations
            )
            self._db.save_learning_metadata("last_save", datetime.now().isoformat())

            logger.debug(f"Knowledge saved to SQLite ({len(self._patterns)} patterns)")
        except Exception as e:
            logger.error(f"Failed to save knowledge to database: {e}")

    def _load_knowledge(self):
        """Load previously learned knowledge from SQLite database"""
        try:
            # Load patterns from database
            patterns_data = self._db.load_learning_patterns()

            for p in patterns_data:
                self._patterns[p["pattern_id"]] = Pattern(
                    pattern_id=p["pattern_id"],
                    pattern_type=p["pattern_type"],
                    conditions=p["conditions"],
                    action=p["action"],
                    expected_outcome=p["expected_outcome"],
                    confidence=p["confidence"],
                    usage_count=p.get("usage_count", 0),
                    success_count=p.get("success_count", 0),
                    last_used=p.get("last_used"),
                )

            # Load metadata
            self._learning_iterations = self._db.load_learning_metadata(
                "learning_iterations", 0
            )

            if self._patterns:
                logger.info(f"Loaded {len(self._patterns)} patterns from SQLite")
        except Exception as e:
            logger.error(f"Failed to load knowledge from database: {e}")


# Singleton instance
_learner: Optional[RecursiveLearner] = None


def get_recursive_learner() -> RecursiveLearner:
    """Get or create the recursive learner singleton"""
    global _learner
    if _learner is None:
        _learner = RecursiveLearner()
    return _learner
