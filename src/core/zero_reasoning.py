"""
Structured Reasoning Orchestrator (Prompt Engineering Framework)

NOTE ON TERMINOLOGY: This module is named "zero_reasoning" but it is NOT a
symbolic logic engine or theorem prover. It does NOT enable the LLM to reason
beyond its base capabilities.

WHAT THIS ACTUALLY DOES:
- Orchestrates structured prompt engineering for LLMs
- Implements Chain-of-Thought (CoT) prompting to force step-by-step reasoning
- Implements Tree-of-Thought (ToT) for exploring multiple reasoning paths
- Maintains a "knowledge base" (dictionary of assertions, not a formal KB)
- Parses JSON responses and manages reasoning state machines

This improves EXPLAINABILITY by forcing the model to show its work, which can
reduce hallucination and improve answer quality. It does NOT add reasoning
capabilities the underlying LLM doesn't already have.

Reference: Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large
Language Models" (2022)
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from enum import Enum
from datetime import datetime
import json
import hashlib
import re

logger = logging.getLogger(__name__)


class ReasoningMode(Enum):
    """Reasoning modes"""

    DEDUCTIVE = "deductive"  # General to specific
    INDUCTIVE = "inductive"  # Specific to general
    ABDUCTIVE = "abductive"  # Best explanation
    ANALOGICAL = "analogical"  # By comparison
    CAUSAL = "causal"  # Cause and effect
    COUNTERFACTUAL = "counterfactual"  # What if


class ConfidenceLevel(Enum):
    """Confidence levels for conclusions"""

    CERTAIN = 1.0
    HIGH = 0.85
    MEDIUM = 0.65
    LOW = 0.45
    UNCERTAIN = 0.25
    UNKNOWN = 0.0


@dataclass
class Premise:
    """A logical premise"""

    premise_id: str
    statement: str
    confidence: float = 1.0
    source: str = "axiom"
    dependencies: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "premiseId": self.premise_id,
            "statement": self.statement,
            "confidence": self.confidence,
            "source": self.source,
            "dependencies": self.dependencies,
        }


@dataclass
class Conclusion:
    """A derived conclusion"""

    conclusion_id: str
    statement: str
    confidence: float
    supporting_premises: List[str]
    reasoning_chain: List[str]
    mode: ReasoningMode
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conclusionId": self.conclusion_id,
            "statement": self.statement,
            "confidence": self.confidence,
            "supportingPremises": self.supporting_premises,
            "reasoningChain": self.reasoning_chain,
            "mode": self.mode.value,
            "timestamp": self.timestamp,
        }


@dataclass
class ReasoningStep:
    """A single step in reasoning"""

    step_id: str
    step_number: int
    thought: str
    action: str
    observation: str
    confidence: float
    alternatives: List[str] = field(default_factory=list)


@dataclass
class ReasoningTree:
    """Tree structure for tree-of-thought reasoning"""

    root_question: str
    branches: List["ReasoningBranch"] = field(default_factory=list)
    best_path: Optional[List[str]] = None
    final_answer: Optional[str] = None
    confidence: float = 0.0


@dataclass
class ReasoningBranch:
    """A branch in the reasoning tree"""

    branch_id: str
    parent_id: Optional[str]
    thought: str
    children: List["ReasoningBranch"] = field(default_factory=list)
    score: float = 0.0
    is_terminal: bool = False


class KnowledgeBase:
    """Dynamic knowledge base for reasoning"""

    def __init__(self):
        self._premises: Dict[str, Premise] = {}
        self._conclusions: Dict[str, Conclusion] = {}
        self._axioms: Set[str] = set()
        self._relations: Dict[str, List[Tuple[str, str]]] = (
            {}
        )  # entity -> [(relation, entity)]

    def add_axiom(self, statement: str) -> str:
        """Add a fundamental axiom"""
        premise_id = hashlib.md5(statement.encode()).hexdigest()[:12]
        premise = Premise(
            premise_id=premise_id, statement=statement, confidence=1.0, source="axiom"
        )
        self._premises[premise_id] = premise
        self._axioms.add(premise_id)
        return premise_id

    def add_premise(
        self,
        statement: str,
        confidence: float = 1.0,
        source: str = "derived",
        dependencies: List[str] = None,
    ) -> str:
        """Add a premise to the knowledge base"""
        premise_id = hashlib.md5(
            f"{statement}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]
        premise = Premise(
            premise_id=premise_id,
            statement=statement,
            confidence=confidence,
            source=source,
            dependencies=dependencies or [],
        )
        self._premises[premise_id] = premise
        return premise_id

    def add_relation(self, entity1: str, relation: str, entity2: str):
        """Add a relation between entities"""
        if entity1 not in self._relations:
            self._relations[entity1] = []
        self._relations[entity1].append((relation, entity2))

    def get_premise(self, premise_id: str) -> Optional[Premise]:
        return self._premises.get(premise_id)

    def get_all_premises(self) -> List[Premise]:
        return list(self._premises.values())

    def find_related(
        self, entity: str, relation: Optional[str] = None
    ) -> List[Tuple[str, str]]:
        """Find related entities"""
        relations = self._relations.get(entity, [])
        if relation:
            return [(r, e) for r, e in relations if r == relation]
        return relations

    def query(self, pattern: str) -> List[Premise]:
        """Query premises matching pattern"""
        results = []
        for premise in self._premises.values():
            if pattern.lower() in premise.statement.lower():
                results.append(premise)
        return results


class ChainOfThought:
    """Chain-of-Thought reasoning implementation"""

    def __init__(self, llm_client: Any = None, model_name: str = "gpt-4"):
        self.llm_client = llm_client
        self.model_name = model_name
        self._reasoning_history: List[List[ReasoningStep]] = []

    async def reason(
        self, question: str, context: str = "", max_steps: int = 10
    ) -> Tuple[str, List[ReasoningStep], float]:
        """Perform chain-of-thought reasoning"""
        steps = []
        current_thought = question

        for i in range(max_steps):
            step = ReasoningStep(
                step_id=f"step_{i}",
                step_number=i + 1,
                thought="",
                action="",
                observation="",
                confidence=0.0,
            )

            # Generate thought
            if self.llm_client:
                thought_prompt = self._build_thought_prompt(question, context, steps)
                thought_response = await self._query_llm(thought_prompt)
                step.thought = thought_response.get("thought", "")
                step.action = thought_response.get("action", "")
                step.observation = thought_response.get("observation", "")
                step.confidence = thought_response.get("confidence", 0.5)
            else:
                # Fallback: structured decomposition
                step.thought = f"Analyzing aspect {i+1} of: {current_thought}"
                step.action = "decompose"
                step.observation = f"Sub-component identified"
                step.confidence = 0.7

            steps.append(step)

            # Check for termination
            if self._is_conclusion_reached(step):
                break

        # Synthesize final answer
        final_answer = self._synthesize_answer(steps)
        overall_confidence = self._calculate_confidence(steps)

        self._reasoning_history.append(steps)

        return final_answer, steps, overall_confidence

    def _build_thought_prompt(
        self, question: str, context: str, steps: List[ReasoningStep]
    ) -> str:
        """Build prompt for thought generation"""
        prompt = f"""Question: {question}

Context: {context}

Previous reasoning steps:
"""
        for step in steps:
            prompt += f"\nStep {step.step_number}: {step.thought}"
            if step.observation:
                prompt += f"\nObservation: {step.observation}"

        prompt += """

Based on the above, provide the next reasoning step in JSON format:
{
    "thought": "your next thought",
    "action": "what action to take",
    "observation": "what you observe",
    "confidence": 0.0-1.0,
    "is_final": true/false
}"""
        return prompt

    async def _query_llm(self, prompt: str) -> Dict[str, Any]:
        """Query LLM for reasoning step"""
        if not self.llm_client:
            return {
                "thought": "Proceeding with logical analysis",
                "action": "analyze",
                "observation": "Pattern identified",
                "confidence": 0.7,
                "is_final": False,
            }

        try:
            response = await self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            content = response.choices[0].message.content
            # Parse JSON from response
            json_match = re.search(r"\{[^{}]*\}", content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {"thought": content, "confidence": 0.5}
        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            return {"thought": "Error in reasoning", "confidence": 0.1}

    def _is_conclusion_reached(self, step: ReasoningStep) -> bool:
        """Check if reasoning has reached a conclusion"""
        conclusion_indicators = ["therefore", "conclude", "final", "answer is"]
        return any(ind in step.thought.lower() for ind in conclusion_indicators)

    def _synthesize_answer(self, steps: List[ReasoningStep]) -> str:
        """Synthesize final answer from steps"""
        if not steps:
            return "Unable to reach conclusion"

        # Combine observations
        observations = [s.observation for s in steps if s.observation]
        thoughts = [s.thought for s in steps if s.thought]

        if observations:
            return f"Based on analysis: {'. '.join(observations[-3:])}"
        elif thoughts:
            return f"Conclusion: {thoughts[-1]}"
        return "Inconclusive"

    def _calculate_confidence(self, steps: List[ReasoningStep]) -> float:
        """Calculate overall confidence"""
        if not steps:
            return 0.0
        confidences = [s.confidence for s in steps]
        # Use geometric mean for confidence propagation
        import math

        return math.exp(sum(math.log(c + 0.01) for c in confidences) / len(confidences))


class TreeOfThought:
    """Tree-of-Thought reasoning for complex problems"""

    def __init__(
        self,
        llm_client: Any = None,
        model_name: str = "gpt-4",
        branching_factor: int = 3,
        max_depth: int = 5,
    ):
        self.llm_client = llm_client
        self.model_name = model_name
        self.branching_factor = branching_factor
        self.max_depth = max_depth

    async def reason(self, question: str, context: str = "") -> ReasoningTree:
        """Perform tree-of-thought reasoning"""
        tree = ReasoningTree(root_question=question)

        # Generate initial branches
        root_branches = await self._generate_branches(question, context, None, 0)
        tree.branches = root_branches

        # Explore and prune
        await self._explore_tree(tree, context)

        # Find best path
        tree.best_path = self._find_best_path(tree)
        tree.final_answer = self._extract_answer(tree)
        tree.confidence = self._calculate_tree_confidence(tree)

        return tree

    async def _generate_branches(
        self, question: str, context: str, parent_id: Optional[str], depth: int
    ) -> List[ReasoningBranch]:
        """Generate reasoning branches"""
        if depth >= self.max_depth:
            return []

        branches = []

        # Generate multiple thoughts
        for i in range(self.branching_factor):
            branch_id = f"branch_{depth}_{i}"

            if self.llm_client:
                thought = await self._generate_thought(question, context, i)
            else:
                thought = f"Approach {i+1}: Consider {'logical' if i == 0 else 'alternative'} perspective"

            branch = ReasoningBranch(
                branch_id=branch_id,
                parent_id=parent_id,
                thought=thought,
                score=0.5 + (0.1 * (self.branching_factor - i)),  # Initial scoring
                is_terminal=(depth == self.max_depth - 1),
            )
            branches.append(branch)

        return branches

    async def _generate_thought(
        self, question: str, context: str, variation: int
    ) -> str:
        """Generate a thought variation"""
        if not self.llm_client:
            return f"Analytical approach {variation + 1}"

        try:
            prompt = f"""Question: {question}
Context: {context}
Generate thought variation {variation + 1} of {self.branching_factor} for solving this:"""

            response = await self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7 + (variation * 0.1),
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Thought generation failed: {e}")
            return f"Approach {variation + 1}"

    async def _explore_tree(self, tree: ReasoningTree, context: str):
        """Explore and expand tree branches"""
        for branch in tree.branches:
            # Score branch
            branch.score = await self._evaluate_branch(branch, tree.root_question)

            # Expand promising branches
            if branch.score > 0.5 and not branch.is_terminal:
                branch.children = await self._generate_branches(
                    tree.root_question,
                    context + f"\nPrevious thought: {branch.thought}",
                    branch.branch_id,
                    1,  # Depth
                )

                # Recursively explore children
                for child in branch.children:
                    child.score = await self._evaluate_branch(child, tree.root_question)

    async def _evaluate_branch(self, branch: ReasoningBranch, question: str) -> float:
        """Evaluate a branch's promise"""
        if self.llm_client:
            try:
                prompt = f"""Rate how well this thought addresses the question (0-1):
Question: {question}
Thought: {branch.thought}
Return only a number between 0 and 1."""

                response = await self.llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                )
                content = response.choices[0].message.content.strip()
                return float(re.search(r"[\d.]+", content).group())
            except:
                pass
        return branch.score

    def _find_best_path(self, tree: ReasoningTree) -> List[str]:
        """Find the best reasoning path"""
        best_path = []
        best_score = 0.0

        def traverse(branches: List[ReasoningBranch], path: List[str], score: float):
            nonlocal best_path, best_score

            for branch in branches:
                new_path = path + [branch.thought]
                new_score = score + branch.score

                if branch.is_terminal or not branch.children:
                    if new_score > best_score:
                        best_score = new_score
                        best_path = new_path
                else:
                    traverse(branch.children, new_path, new_score)

        traverse(tree.branches, [], 0.0)
        return best_path

    def _extract_answer(self, tree: ReasoningTree) -> str:
        """Extract final answer from best path"""
        if not tree.best_path:
            return "No conclusion reached"
        return f"Conclusion based on reasoning: {tree.best_path[-1]}"

    def _calculate_tree_confidence(self, tree: ReasoningTree) -> float:
        """Calculate confidence from tree exploration"""
        if not tree.branches:
            return 0.0

        max_scores = []
        for branch in tree.branches:
            max_scores.append(branch.score)
            if branch.children:
                max_scores.append(max(c.score for c in branch.children))

        return sum(max_scores) / len(max_scores) if max_scores else 0.0


class MetaCognition:
    """Meta-cognitive layer for self-aware reasoning"""

    def __init__(self):
        self._reasoning_quality: List[float] = []
        self._error_patterns: Dict[str, int] = {}
        self._successful_strategies: Dict[str, int] = {}

    def evaluate_reasoning(
        self, steps: List[ReasoningStep], actual_result: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate reasoning quality"""
        evaluation = {
            "step_count": len(steps),
            "avg_confidence": (
                sum(s.confidence for s in steps) / len(steps) if steps else 0
            ),
            "coherence": self._check_coherence(steps),
            "completeness": self._check_completeness(steps),
            "suggestions": [],
        }

        if evaluation["coherence"] < 0.5:
            evaluation["suggestions"].append("Consider more systematic approach")
        if evaluation["completeness"] < 0.5:
            evaluation["suggestions"].append("Explore more alternatives")

        self._reasoning_quality.append(evaluation["avg_confidence"])

        return evaluation

    def _check_coherence(self, steps: List[ReasoningStep]) -> float:
        """Check logical coherence between steps"""
        if len(steps) < 2:
            return 1.0

        # Simple coherence check: confidence should not drop dramatically
        drops = 0
        for i in range(1, len(steps)):
            if steps[i].confidence < steps[i - 1].confidence - 0.3:
                drops += 1

        return 1.0 - (drops / len(steps))

    def _check_completeness(self, steps: List[ReasoningStep]) -> float:
        """Check if reasoning covered all aspects"""
        if not steps:
            return 0.0

        # Check for diversity in actions
        actions = set(s.action for s in steps if s.action)
        return min(len(actions) / 3, 1.0)  # Expect at least 3 different actions

    def record_strategy(self, strategy: str, success: bool):
        """Record strategy usage"""
        if success:
            self._successful_strategies[strategy] = (
                self._successful_strategies.get(strategy, 0) + 1
            )
        else:
            self._error_patterns[strategy] = self._error_patterns.get(strategy, 0) + 1

    def get_recommended_strategy(self) -> Optional[str]:
        """Get most successful strategy"""
        if not self._successful_strategies:
            return None
        return max(self._successful_strategies, key=self._successful_strategies.get)


class AbsoluteZeroReasoner:
    """Main reasoning engine combining all approaches"""

    def __init__(self, llm_client: Any = None, model_name: str = "gpt-4"):
        self.knowledge_base = KnowledgeBase()
        self.cot = ChainOfThought(llm_client, model_name)
        self.tot = TreeOfThought(llm_client, model_name)
        self.meta = MetaCognition()
        self.llm_client = llm_client
        self.model_name = model_name

        # Initialize with fundamental axioms
        self._init_axioms()

    def _init_axioms(self):
        """Initialize fundamental axioms"""
        axioms = [
            "If A implies B and B implies C, then A implies C",
            "Something cannot be both true and false simultaneously",
            "Every effect has a cause",
            "The simplest explanation is often correct",
            "Patterns in data suggest underlying structure",
        ]
        for axiom in axioms:
            self.knowledge_base.add_axiom(axiom)

    async def reason(
        self,
        question: str,
        context: str = "",
        mode: ReasoningMode = ReasoningMode.DEDUCTIVE,
    ) -> Dict[str, Any]:
        """Perform reasoning using the appropriate method"""

        # Select reasoning strategy
        if mode == ReasoningMode.DEDUCTIVE:
            answer, steps, confidence = await self.cot.reason(question, context)
            reasoning_type = "chain_of_thought"
        elif self._is_complex_question(question):
            tree = await self.tot.reason(question, context)
            answer = tree.final_answer
            steps = []  # Tree doesn't use linear steps
            confidence = tree.confidence
            reasoning_type = "tree_of_thought"
        else:
            answer, steps, confidence = await self.cot.reason(question, context)
            reasoning_type = "chain_of_thought"

        # Meta-cognitive evaluation
        if steps:
            evaluation = self.meta.evaluate_reasoning(steps)
        else:
            evaluation = {"suggestions": []}

        # Record conclusion
        if confidence > 0.5:
            conclusion = Conclusion(
                conclusion_id=hashlib.md5(f"{question}{answer}".encode()).hexdigest()[
                    :12
                ],
                statement=answer,
                confidence=confidence,
                supporting_premises=[],
                reasoning_chain=[s.thought for s in steps] if steps else [],
                mode=mode,
            )
            self.knowledge_base._conclusions[conclusion.conclusion_id] = conclusion

        return {
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "reasoning_type": reasoning_type,
            "steps": (
                [{"thought": s.thought, "confidence": s.confidence} for s in steps]
                if steps
                else []
            ),
            "evaluation": evaluation,
            "timestamp": datetime.now().isoformat(),
        }

    def _is_complex_question(self, question: str) -> bool:
        """Determine if question requires tree-of-thought"""
        complex_indicators = [
            "compare",
            "multiple",
            "different ways",
            "alternatives",
            "best approach",
        ]
        return any(ind in question.lower() for ind in complex_indicators)

    async def reason_from_first_principles(self, problem: str) -> Dict[str, Any]:
        """Reason from first principles without prior examples"""
        # Decompose problem
        decomposition = await self._decompose_problem(problem)

        # Build up from axioms
        reasoning_chain = []
        current_understanding = ""

        for component in decomposition:
            # Find relevant axioms
            relevant = self.knowledge_base.query(component)

            # Reason about component
            result = await self.reason(
                f"Given: {current_understanding}\nAnalyze: {component}",
                context="\n".join(p.statement for p in relevant),
            )

            reasoning_chain.append(
                {
                    "component": component,
                    "analysis": result["answer"],
                    "confidence": result["confidence"],
                }
            )

            current_understanding += f"\n{result['answer']}"

        # Synthesize final answer
        synthesis = await self.reason(
            f"Synthesize understanding of: {problem}", context=current_understanding
        )

        return {
            "problem": problem,
            "decomposition": decomposition,
            "reasoning_chain": reasoning_chain,
            "synthesis": synthesis["answer"],
            "overall_confidence": synthesis["confidence"],
        }

    async def _decompose_problem(self, problem: str) -> List[str]:
        """Decompose problem into components"""
        if self.llm_client:
            try:
                prompt = f"""Decompose this problem into fundamental components:
{problem}

Return a JSON list of components:"""
                response = await self.llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                content = response.choices[0].message.content
                match = re.search(r"\[.*\]", content, re.DOTALL)
                if match:
                    return json.loads(match.group())
            except:
                pass

        # Fallback decomposition
        return [problem]


# Factory function
def create_reasoner(
    llm_client: Any = None, model_name: str = "gpt-4"
) -> AbsoluteZeroReasoner:
    """Create a reasoning engine"""
    return AbsoluteZeroReasoner(llm_client, model_name)
