"""
Refactory Feedback Loop System
Enterprise-grade implementation for automated code refactoring based on quality feedback.
"""

import asyncio
import logging
import ast
import re
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from enum import Enum
from datetime import datetime
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class RefactorType(Enum):
    """Types of refactoring operations"""

    EXTRACT_METHOD = "extract_method"
    INLINE_METHOD = "inline_method"
    RENAME = "rename"
    MOVE = "move"
    EXTRACT_VARIABLE = "extract_variable"
    INLINE_VARIABLE = "inline_variable"
    SIMPLIFY_CONDITIONAL = "simplify_conditional"
    DECOMPOSE_CONDITIONAL = "decompose_conditional"
    CONSOLIDATE_DUPLICATE = "consolidate_duplicate"
    REMOVE_DEAD_CODE = "remove_dead_code"
    OPTIMIZE_IMPORTS = "optimize_imports"
    ADD_TYPE_HINTS = "add_type_hints"
    IMPROVE_NAMING = "improve_naming"


class QualityDimension(Enum):
    """Code quality dimensions"""

    READABILITY = "readability"
    MAINTAINABILITY = "maintainability"
    PERFORMANCE = "performance"
    TESTABILITY = "testability"
    SECURITY = "security"
    COMPLEXITY = "complexity"


@dataclass
class CodeMetrics:
    """Code quality metrics"""

    lines_of_code: int = 0
    cyclomatic_complexity: int = 0
    cognitive_complexity: int = 0
    duplicate_lines: int = 0
    test_coverage: float = 0.0
    documentation_ratio: float = 0.0
    type_hint_coverage: float = 0.0
    avg_function_length: float = 0.0
    max_function_length: int = 0
    import_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "linesOfCode": self.lines_of_code,
            "cyclomaticComplexity": self.cyclomatic_complexity,
            "cognitiveComplexity": self.cognitive_complexity,
            "duplicateLines": self.duplicate_lines,
            "testCoverage": self.test_coverage,
            "documentationRatio": self.documentation_ratio,
            "typeHintCoverage": self.type_hint_coverage,
            "avgFunctionLength": self.avg_function_length,
            "maxFunctionLength": self.max_function_length,
            "importCount": self.import_count,
        }

    def quality_score(self) -> float:
        """Calculate overall quality score"""
        scores = []

        # Complexity (lower is better)
        complexity_score = max(0, 1 - (self.cyclomatic_complexity / 50))
        scores.append(complexity_score)

        # Documentation (higher is better)
        scores.append(self.documentation_ratio)

        # Type hints (higher is better)
        scores.append(self.type_hint_coverage)

        # Function length (shorter is better)
        length_score = max(0, 1 - (self.avg_function_length / 50))
        scores.append(length_score)

        return sum(scores) / len(scores) if scores else 0.0


@dataclass
class RefactorSuggestion:
    """A refactoring suggestion"""

    suggestion_id: str
    refactor_type: RefactorType
    target: str  # Function/class/variable name
    location: Tuple[int, int]  # Line, column
    description: str
    impact: Dict[QualityDimension, float]  # Expected improvement
    priority: float  # 0-1, higher is more important
    code_before: Optional[str] = None
    code_after: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suggestionId": self.suggestion_id,
            "refactorType": self.refactor_type.value,
            "target": self.target,
            "location": {"line": self.location[0], "column": self.location[1]},
            "description": self.description,
            "impact": {k.value: v for k, v in self.impact.items()},
            "priority": self.priority,
            "codeBefore": self.code_before,
            "codeAfter": self.code_after,
        }


@dataclass
class FeedbackItem:
    """Feedback on code quality"""

    feedback_id: str
    source: str  # "static_analysis", "runtime", "user", "test"
    dimension: QualityDimension
    message: str
    severity: float  # 0-1
    location: Optional[Tuple[int, int]] = None
    suggestions: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class CodeAnalyzer:
    """Analyzes code for quality metrics and issues"""

    def __init__(self):
        self._complexity_weights = {
            "if": 1,
            "elif": 1,
            "for": 1,
            "while": 1,
            "except": 1,
            "and": 1,
            "or": 1,
        }

    def analyze(self, code: str) -> CodeMetrics:
        """Analyze code and return metrics"""
        metrics = CodeMetrics()

        lines = code.split("\n")
        metrics.lines_of_code = len(
            [l for l in lines if l.strip() and not l.strip().startswith("#")]
        )

        try:
            tree = ast.parse(code)
            metrics.cyclomatic_complexity = self._calculate_cyclomatic(tree)
            metrics.cognitive_complexity = self._calculate_cognitive(tree)
            metrics.documentation_ratio = self._calculate_doc_ratio(tree)
            metrics.type_hint_coverage = self._calculate_type_hints(tree)

            func_lengths = self._get_function_lengths(tree)
            if func_lengths:
                metrics.avg_function_length = sum(func_lengths) / len(func_lengths)
                metrics.max_function_length = max(func_lengths)

            metrics.import_count = self._count_imports(tree)
        except SyntaxError:
            pass

        metrics.duplicate_lines = self._find_duplicates(lines)

        return metrics

    def _calculate_cyclomatic(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1

        return complexity

    def _calculate_cognitive(self, tree: ast.AST) -> int:
        """Calculate cognitive complexity"""
        complexity = 0
        nesting = 0

        class CognitiveVisitor(ast.NodeVisitor):
            def __init__(self):
                self.complexity = 0
                self.nesting = 0

            def visit_If(self, node):
                self.complexity += 1 + self.nesting
                self.nesting += 1
                self.generic_visit(node)
                self.nesting -= 1

            def visit_For(self, node):
                self.complexity += 1 + self.nesting
                self.nesting += 1
                self.generic_visit(node)
                self.nesting -= 1

            def visit_While(self, node):
                self.complexity += 1 + self.nesting
                self.nesting += 1
                self.generic_visit(node)
                self.nesting -= 1

            def visit_BoolOp(self, node):
                self.complexity += len(node.values) - 1
                self.generic_visit(node)

        visitor = CognitiveVisitor()
        visitor.visit(tree)
        return visitor.complexity

    def _calculate_doc_ratio(self, tree: ast.AST) -> float:
        """Calculate documentation coverage"""
        total = 0
        documented = 0

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                total += 1
                if ast.get_docstring(node):
                    documented += 1

        return documented / total if total > 0 else 1.0

    def _calculate_type_hints(self, tree: ast.AST) -> float:
        """Calculate type hint coverage"""
        total_args = 0
        hinted_args = 0

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for arg in node.args.args:
                    total_args += 1
                    if arg.annotation:
                        hinted_args += 1

                if node.returns:
                    hinted_args += 1
                total_args += 1  # For return type

        return hinted_args / total_args if total_args > 0 else 1.0

    def _get_function_lengths(self, tree: ast.AST) -> List[int]:
        """Get lengths of all functions"""
        lengths = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if hasattr(node, "end_lineno") and hasattr(node, "lineno"):
                    lengths.append(node.end_lineno - node.lineno + 1)

        return lengths

    def _count_imports(self, tree: ast.AST) -> int:
        """Count import statements"""
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                count += 1
        return count

    def _find_duplicates(self, lines: List[str]) -> int:
        """Find duplicate lines"""
        seen = {}
        duplicates = 0

        for line in lines:
            line = line.strip()
            if line and len(line) > 10:  # Ignore short lines
                if line in seen:
                    duplicates += 1
                seen[line] = True

        return duplicates


class RefactorEngine:
    """Engine for generating and applying refactorings"""

    def __init__(self, llm_client: Any = None, model_name: str = "gpt-4"):
        self.llm_client = llm_client
        self.model_name = model_name
        self.analyzer = CodeAnalyzer()

    def suggest_refactorings(self, code: str) -> List[RefactorSuggestion]:
        """Generate refactoring suggestions"""
        suggestions = []
        metrics = self.analyzer.analyze(code)

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return suggestions

        # Check for long functions
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_len = getattr(node, "end_lineno", 0) - getattr(node, "lineno", 0)
                if func_len > 30:
                    suggestions.append(
                        RefactorSuggestion(
                            suggestion_id=hashlib.md5(
                                f"{node.name}_{node.lineno}".encode()
                            ).hexdigest()[:8],
                            refactor_type=RefactorType.EXTRACT_METHOD,
                            target=node.name,
                            location=(node.lineno, node.col_offset),
                            description=f"Function '{node.name}' is {func_len} lines. Consider extracting smaller methods.",
                            impact={
                                QualityDimension.READABILITY: 0.3,
                                QualityDimension.MAINTAINABILITY: 0.2,
                            },
                            priority=min(func_len / 100, 0.9),
                        )
                    )

        # Check for complex conditionals
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                condition_complexity = self._count_boolean_ops(node.test)
                if condition_complexity > 3:
                    suggestions.append(
                        RefactorSuggestion(
                            suggestion_id=hashlib.md5(
                                f"if_{node.lineno}".encode()
                            ).hexdigest()[:8],
                            refactor_type=RefactorType.DECOMPOSE_CONDITIONAL,
                            target="conditional",
                            location=(node.lineno, node.col_offset),
                            description=f"Complex conditional with {condition_complexity} boolean operations.",
                            impact={
                                QualityDimension.READABILITY: 0.2,
                                QualityDimension.COMPLEXITY: 0.15,
                            },
                            priority=min(condition_complexity / 10, 0.8),
                        )
                    )

        # Check for missing type hints
        if metrics.type_hint_coverage < 0.7:
            suggestions.append(
                RefactorSuggestion(
                    suggestion_id=hashlib.md5(
                        f"types_{datetime.now()}".encode()
                    ).hexdigest()[:8],
                    refactor_type=RefactorType.ADD_TYPE_HINTS,
                    target="module",
                    location=(1, 0),
                    description=f"Type hint coverage is {metrics.type_hint_coverage:.0%}. Add type hints to improve code quality.",
                    impact={
                        QualityDimension.MAINTAINABILITY: 0.25,
                        QualityDimension.TESTABILITY: 0.15,
                    },
                    priority=0.6,
                )
            )

        # Check for missing documentation
        if metrics.documentation_ratio < 0.5:
            suggestions.append(
                RefactorSuggestion(
                    suggestion_id=hashlib.md5(
                        f"docs_{datetime.now()}".encode()
                    ).hexdigest()[:8],
                    refactor_type=RefactorType.IMPROVE_NAMING,
                    target="module",
                    location=(1, 0),
                    description=f"Documentation coverage is {metrics.documentation_ratio:.0%}. Add docstrings.",
                    impact={
                        QualityDimension.READABILITY: 0.2,
                        QualityDimension.MAINTAINABILITY: 0.2,
                    },
                    priority=0.5,
                )
            )

        # Sort by priority
        suggestions.sort(key=lambda s: s.priority, reverse=True)

        return suggestions

    def _count_boolean_ops(self, node: ast.AST) -> int:
        """Count boolean operations in expression"""
        count = 0
        for child in ast.walk(node):
            if isinstance(child, ast.BoolOp):
                count += len(child.values)
        return count

    async def apply_refactoring(
        self, code: str, suggestion: RefactorSuggestion
    ) -> Optional[str]:
        """Apply a refactoring suggestion"""
        if self.llm_client:
            return await self._llm_refactor(code, suggestion)
        return self._rule_based_refactor(code, suggestion)

    async def _llm_refactor(
        self, code: str, suggestion: RefactorSuggestion
    ) -> Optional[str]:
        """Use LLM for refactoring"""
        try:
            prompt = f"""Refactor the following code according to this suggestion:

Suggestion: {suggestion.description}
Refactoring Type: {suggestion.refactor_type.value}
Target: {suggestion.target}

Code:
```python
{code}
```

Return ONLY the refactored code, nothing else."""

            response = await self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )

            content = response.choices[0].message.content
            # Extract code from response
            code_match = re.search(r"```python\n(.*?)\n```", content, re.DOTALL)
            if code_match:
                return code_match.group(1)
            return content
        except Exception as e:
            logger.error(f"LLM refactoring failed: {e}")
            return None

    def _rule_based_refactor(
        self, code: str, suggestion: RefactorSuggestion
    ) -> Optional[str]:
        """Apply rule-based refactoring"""
        # Simple rule-based transformations
        if suggestion.refactor_type == RefactorType.OPTIMIZE_IMPORTS:
            return self._optimize_imports(code)

        # For complex refactorings, return None to indicate LLM needed
        return None

    def _optimize_imports(self, code: str) -> str:
        """Optimize imports"""
        lines = code.split("\n")
        imports = []
        from_imports = []
        other_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("import "):
                imports.append(stripped)
            elif stripped.startswith("from "):
                from_imports.append(stripped)
            else:
                other_lines.append(line)

        # Sort and deduplicate
        imports = sorted(set(imports))
        from_imports = sorted(set(from_imports))

        # Reconstruct
        result = []
        if imports:
            result.extend(imports)
            result.append("")
        if from_imports:
            result.extend(from_imports)
            result.append("")
        result.extend(other_lines)

        return "\n".join(result)


class FeedbackCollector:
    """Collects feedback from various sources"""

    def __init__(self):
        self._feedback: List[FeedbackItem] = []
        self._sources: Dict[str, Callable] = {}

    def register_source(self, name: str, collector: Callable):
        """Register a feedback source"""
        self._sources[name] = collector

    async def collect(self, code: str) -> List[FeedbackItem]:
        """Collect feedback from all sources"""
        feedback = []

        for name, collector in self._sources.items():
            try:
                if asyncio.iscoroutinefunction(collector):
                    items = await collector(code)
                else:
                    items = collector(code)

                for item in items:
                    item.source = name
                    feedback.append(item)
            except Exception as e:
                logger.error(f"Feedback collection from {name} failed: {e}")

        self._feedback.extend(feedback)
        return feedback

    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get summary of collected feedback"""
        by_dimension = {}
        by_severity = {"high": 0, "medium": 0, "low": 0}

        for item in self._feedback:
            dim = item.dimension.value
            by_dimension[dim] = by_dimension.get(dim, 0) + 1

            if item.severity > 0.7:
                by_severity["high"] += 1
            elif item.severity > 0.3:
                by_severity["medium"] += 1
            else:
                by_severity["low"] += 1

        return {
            "total": len(self._feedback),
            "by_dimension": by_dimension,
            "by_severity": by_severity,
        }


class RefactoryFeedbackLoop:
    """Main feedback loop orchestrator"""

    def __init__(self, llm_client: Any = None, model_name: str = "gpt-4"):
        self.analyzer = CodeAnalyzer()
        self.engine = RefactorEngine(llm_client, model_name)
        self.collector = FeedbackCollector()
        self._history: List[Dict[str, Any]] = []

        # Register default feedback source
        self.collector.register_source(
            "static_analysis", self._static_analysis_feedback
        )

    def _static_analysis_feedback(self, code: str) -> List[FeedbackItem]:
        """Generate feedback from static analysis"""
        feedback = []
        metrics = self.analyzer.analyze(code)

        if metrics.cyclomatic_complexity > 20:
            feedback.append(
                FeedbackItem(
                    feedback_id=hashlib.md5(
                        f"cc_{datetime.now()}".encode()
                    ).hexdigest()[:8],
                    source="static_analysis",
                    dimension=QualityDimension.COMPLEXITY,
                    message=f"High cyclomatic complexity: {metrics.cyclomatic_complexity}",
                    severity=min(metrics.cyclomatic_complexity / 50, 1.0),
                )
            )

        if metrics.documentation_ratio < 0.5:
            feedback.append(
                FeedbackItem(
                    feedback_id=hashlib.md5(
                        f"doc_{datetime.now()}".encode()
                    ).hexdigest()[:8],
                    source="static_analysis",
                    dimension=QualityDimension.READABILITY,
                    message=f"Low documentation: {metrics.documentation_ratio:.0%}",
                    severity=1 - metrics.documentation_ratio,
                )
            )

        return feedback

    async def run_iteration(
        self, code: str, auto_apply: bool = False
    ) -> Dict[str, Any]:
        """Run one iteration of the feedback loop"""
        iteration_start = datetime.now()

        # Analyze current state
        before_metrics = self.analyzer.analyze(code)
        before_score = before_metrics.quality_score()

        # Collect feedback
        feedback = await self.collector.collect(code)

        # Generate suggestions
        suggestions = self.engine.suggest_refactorings(code)

        # Apply refactorings if enabled
        applied = []
        current_code = code

        if auto_apply and suggestions:
            for suggestion in suggestions[:3]:  # Limit auto-apply
                refactored = await self.engine.apply_refactoring(
                    current_code, suggestion
                )
                if refactored:
                    current_code = refactored
                    applied.append(suggestion.suggestion_id)

        # Analyze after state
        after_metrics = self.analyzer.analyze(current_code)
        after_score = after_metrics.quality_score()

        # Record iteration
        result = {
            "iteration": len(self._history) + 1,
            "timestamp": iteration_start.isoformat(),
            "before_score": before_score,
            "after_score": after_score,
            "improvement": after_score - before_score,
            "feedback_count": len(feedback),
            "suggestion_count": len(suggestions),
            "applied_count": len(applied),
            "metrics_before": before_metrics.to_dict(),
            "metrics_after": after_metrics.to_dict(),
            "suggestions": [s.to_dict() for s in suggestions[:5]],
            "code": current_code if auto_apply else None,
        }

        self._history.append(result)

        return result

    def get_improvement_trend(self) -> List[float]:
        """Get quality score trend over iterations"""
        return [h.get("after_score", 0) for h in self._history]

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of feedback loop"""
        if not self._history:
            return {"iterations": 0}

        scores = self.get_improvement_trend()

        return {
            "iterations": len(self._history),
            "initial_score": self._history[0]["before_score"],
            "current_score": scores[-1] if scores else 0,
            "total_improvement": (
                scores[-1] - self._history[0]["before_score"] if scores else 0
            ),
            "total_suggestions": sum(
                h.get("suggestion_count", 0) for h in self._history
            ),
            "total_applied": sum(h.get("applied_count", 0) for h in self._history),
            "feedback_summary": self.collector.get_feedback_summary(),
        }


# Singleton instance
_feedback_loop: Optional[RefactoryFeedbackLoop] = None


def get_feedback_loop(llm_client: Any = None) -> RefactoryFeedbackLoop:
    """Get or create the feedback loop singleton"""
    global _feedback_loop
    if _feedback_loop is None:
        _feedback_loop = RefactoryFeedbackLoop(llm_client)
    return _feedback_loop
