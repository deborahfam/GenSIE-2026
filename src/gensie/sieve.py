"""
SIEVE v2 — Schema-Informed Extraction with Verification and Enhancement.

Contest-ready pipelines: sieve-fast, sieve-verified, sieve-fewshot.
Model-agnostic, deterministic repair, grounding guards, schema-valid fallbacks.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from openai import OpenAI
from gensie.agent import GenSIEAgent
from gensie.task import Task
from dotenv import load_dotenv
from logging import getLogger

load_dotenv()
logger = getLogger("gensie")

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

_CLIENT: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OpenAI(
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY", "sk-dummy"),
            timeout=55.0,
        )
    return _CLIENT


# ---------------------------------------------------------------------------
# Schema Planner
# ---------------------------------------------------------------------------

@dataclass
class FieldPlan:
    """Plan for a single field in the schema."""
    name: str
    path: str  # dot-separated path e.g. "side_effects.*.reaction"
    field_type: str  # "string", "number", "integer", "boolean", "array", "object", "enum", "null"
    nullable: bool = False
    required: bool = False
    enum_values: Optional[List[str]] = None
    description: str = ""
    default: Any = None
    has_default: bool = False
    items_plan: Optional["SchemaPlan"] = None  # for arrays of objects
    items_type: Optional[str] = None  # for arrays of primitives
    nested_plan: Optional["SchemaPlan"] = None  # for nested objects
    constraints: List[str] = field(default_factory=list)


@dataclass
class SchemaPlan:
    """Complete plan for a JSON schema."""
    title: str = ""
    fields: List[FieldPlan] = field(default_factory=list)
    required_names: List[str] = field(default_factory=list)
    all_names: List[str] = field(default_factory=list)


def build_schema_plan(schema: Dict[str, Any], defs: Optional[Dict[str, Any]] = None, path_prefix: str = "") -> SchemaPlan:
    """Recursively plans a JSON schema into FieldPlans."""
    if defs is None:
        defs = schema.get("$defs", {})

    plan = SchemaPlan(
        title=schema.get("title", ""),
        required_names=schema.get("required", []),
        all_names=list(schema.get("properties", {}).keys()),
    )

    properties = schema.get("properties", {})
    required_set = set(plan.required_names)

    for fname, fdef in properties.items():
        fp = _plan_field(fname, fdef, defs, path_prefix, fname in required_set)
        plan.fields.append(fp)

    return plan


def _resolve_ref(ref: str, defs: Dict[str, Any]) -> Dict[str, Any]:
    """Resolves a $ref like '#/$defs/SoftwareType' to its definition."""
    parts = ref.replace("#/", "").split("/")
    curr: Any = {"$defs": defs}
    for p in parts:
        if isinstance(curr, dict):
            curr = curr.get(p, {})
        else:
            return {}
    return curr if isinstance(curr, dict) else {}


def _plan_field(name: str, fdef: Dict[str, Any], defs: Dict[str, Any], path_prefix: str, is_required: bool) -> FieldPlan:
    """Plans a single field definition."""
    path = f"{path_prefix}.{name}" if path_prefix else name
    description = fdef.get("description", "")
    nullable = False
    has_default = "default" in fdef
    default = fdef.get("default")

    # Resolve $ref at top level
    if "$ref" in fdef and "anyOf" not in fdef:
        resolved = _resolve_ref(fdef["$ref"], defs)
        if resolved.get("enum"):
            return FieldPlan(
                name=name, path=path, field_type="enum", nullable=False,
                required=is_required, enum_values=resolved["enum"],
                description=description or resolved.get("description", ""),
                has_default=has_default, default=default,
            )
        # If it's an object ref, recurse
        if resolved.get("type") == "object":
            nested = build_schema_plan(resolved, defs, path)
            return FieldPlan(
                name=name, path=path, field_type="object", nullable=False,
                required=is_required, description=description,
                nested_plan=nested, has_default=has_default, default=default,
            )

    # Handle anyOf (nullable unions)
    if "anyOf" in fdef:
        branches = fdef["anyOf"]
        null_branch = any(b.get("type") == "null" for b in branches)
        nullable = null_branch
        # Find the non-null branch
        non_null = [b for b in branches if b.get("type") != "null"]
        if non_null:
            inner = non_null[0]
            # Resolve ref inside anyOf
            if "$ref" in inner:
                resolved = _resolve_ref(inner["$ref"], defs)
                if resolved.get("enum"):
                    return FieldPlan(
                        name=name, path=path, field_type="enum", nullable=nullable,
                        required=is_required, enum_values=resolved["enum"],
                        description=description or resolved.get("description", ""),
                        has_default=has_default, default=default,
                    )
                if resolved.get("type") == "object":
                    nested = build_schema_plan(resolved, defs, path)
                    return FieldPlan(
                        name=name, path=path, field_type="object", nullable=nullable,
                        required=is_required, description=description,
                        nested_plan=nested, has_default=has_default, default=default,
                    )
            inner_type = inner.get("type", "string")
            if inner_type == "array":
                items_plan, items_type = _plan_items(inner.get("items", {}), defs, path)
                return FieldPlan(
                    name=name, path=path, field_type="array", nullable=nullable,
                    required=is_required, description=description,
                    items_plan=items_plan, items_type=items_type,
                    has_default=has_default, default=default,
                )
            if inner_type == "object":
                nested = build_schema_plan(inner, defs, path)
                return FieldPlan(
                    name=name, path=path, field_type="object", nullable=nullable,
                    required=is_required, description=description,
                    nested_plan=nested, has_default=has_default, default=default,
                )
            return FieldPlan(
                name=name, path=path, field_type=inner_type, nullable=nullable,
                required=is_required, description=description,
                has_default=has_default, default=default,
            )

    # Direct type
    ftype = fdef.get("type", "string")

    if "enum" in fdef:
        return FieldPlan(
            name=name, path=path, field_type="enum", nullable=nullable,
            required=is_required, enum_values=fdef["enum"],
            description=description, has_default=has_default, default=default,
        )

    if ftype == "array":
        items_plan, items_type = _plan_items(fdef.get("items", {}), defs, path)
        return FieldPlan(
            name=name, path=path, field_type="array", nullable=nullable,
            required=is_required, description=description,
            items_plan=items_plan, items_type=items_type,
            has_default=has_default, default=default,
        )

    if ftype == "object":
        nested = build_schema_plan(fdef, defs, path)
        return FieldPlan(
            name=name, path=path, field_type="object", nullable=nullable,
            required=is_required, description=description,
            nested_plan=nested, has_default=has_default, default=default,
        )

    return FieldPlan(
        name=name, path=path, field_type=ftype, nullable=nullable,
        required=is_required, description=description,
        has_default=has_default, default=default,
    )


def _plan_items(items_def: Dict[str, Any], defs: Dict[str, Any], path: str) -> Tuple[Optional[SchemaPlan], Optional[str]]:
    """Plans array items. Returns (items_plan_for_objects, items_type_for_primitives)."""
    if "$ref" in items_def:
        resolved = _resolve_ref(items_def["$ref"], defs)
        if resolved.get("type") == "object":
            return build_schema_plan(resolved, defs, f"{path}.*"), None
        if resolved.get("enum"):
            return None, "enum"
        return None, resolved.get("type", "string")
    if items_def.get("type") == "object":
        return build_schema_plan(items_def, defs, f"{path}.*"), None
    return None, items_def.get("type", "string")


# ---------------------------------------------------------------------------
# Schema-valid fallback (empty output)
# ---------------------------------------------------------------------------

def build_empty(schema: Dict[str, Any], defs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Builds a schema-valid empty/default output. Never returns {"error": ...}."""
    if defs is None:
        defs = schema.get("$defs", {})

    plan = build_schema_plan(schema, defs)
    return _empty_from_plan(plan, defs)


def _empty_from_plan(plan: SchemaPlan, defs: Dict[str, Any]) -> Dict[str, Any]:
    result = {}
    for fp in plan.fields:
        result[fp.name] = _empty_field(fp, defs)
    return result


def _empty_field(fp: FieldPlan, defs: Dict[str, Any]) -> Any:
    if fp.nullable:
        return fp.default if fp.has_default else None
    if fp.has_default and fp.default is not None:
        return fp.default
    if fp.field_type == "string":
        return ""
    if fp.field_type == "number" or fp.field_type == "integer":
        return 0
    if fp.field_type == "boolean":
        return False
    if fp.field_type == "enum":
        if fp.enum_values:
            return fp.enum_values[0]
        return ""
    if fp.field_type == "array":
        return []
    if fp.field_type == "object":
        if fp.nested_plan:
            return _empty_from_plan(fp.nested_plan, defs)
        return {}
    return None


# ---------------------------------------------------------------------------
# Deterministic Repair
# ---------------------------------------------------------------------------

def repair_to_schema(candidate: Any, schema: Dict[str, Any], defs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Deterministically repairs a candidate output to match the schema:
    - Removes extra keys
    - Fills missing required fields with defaults/empty
    - Coerces safe primitives (str->int, str->bool, etc.)
    - Fixes enum casing
    - Repairs arrays (wraps single items)
    """
    if defs is None:
        defs = schema.get("$defs", {})
    if not isinstance(candidate, dict):
        return build_empty(schema, defs)

    plan = build_schema_plan(schema, defs)
    return _repair_object(candidate, plan, defs)


def _repair_object(obj: Any, plan: SchemaPlan, defs: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return _empty_from_plan(plan, defs)

    result = {}
    valid_names = set(plan.all_names)

    for fp in plan.fields:
        if fp.name in obj:
            result[fp.name] = _repair_field(obj[fp.name], fp, defs)
        else:
            # Missing field
            result[fp.name] = _empty_field(fp, defs)

    return result


def _repair_field(value: Any, fp: FieldPlan, defs: Dict[str, Any]) -> Any:
    # Handle null
    if value is None:
        if fp.nullable:
            return None
        return _empty_field(fp, defs)

    if fp.field_type == "string":
        if isinstance(value, str):
            return value
        return str(value)

    if fp.field_type in ("number", "integer"):
        return _coerce_number(value, fp.field_type)

    if fp.field_type == "boolean":
        return _coerce_bool(value)

    if fp.field_type == "enum":
        return _repair_enum(value, fp)

    if fp.field_type == "array":
        return _repair_array(value, fp, defs)

    if fp.field_type == "object":
        if fp.nested_plan:
            return _repair_object(value, fp.nested_plan, defs)
        if isinstance(value, dict):
            return value
        return {}

    return value


def _coerce_number(value: Any, target: str) -> Any:
    if isinstance(value, (int, float)):
        return int(value) if target == "integer" else value
    if isinstance(value, str):
        try:
            v = float(value.replace(",", "."))
            return int(v) if target == "integer" else v
        except (ValueError, TypeError):
            return 0
    return 0


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes", "sí", "si")
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _repair_enum(value: Any, fp: FieldPlan) -> Any:
    if not fp.enum_values:
        return value
    if value in fp.enum_values:
        return value
    # Case-insensitive match
    if isinstance(value, str):
        lower_map = {v.lower(): v for v in fp.enum_values}
        if value.lower() in lower_map:
            return lower_map[value.lower()]
        # Partial match (value contains enum or vice versa)
        for ev in fp.enum_values:
            if ev.lower() in value.lower() or value.lower() in ev.lower():
                return ev
    # If nullable, return None; otherwise first enum value
    if fp.nullable:
        return None
    return fp.enum_values[0]


def _repair_array(value: Any, fp: FieldPlan, defs: Dict[str, Any]) -> List:
    # Wrap single item
    if not isinstance(value, list):
        value = [value] if value is not None else []

    if fp.items_plan:
        # Array of objects
        return [_repair_object(item, fp.items_plan, defs) for item in value if isinstance(item, dict)]
    if fp.items_type == "string":
        return [str(item) for item in value if item is not None]
    if fp.items_type in ("number", "integer"):
        return [_coerce_number(item, fp.items_type) for item in value]
    return value


# ---------------------------------------------------------------------------
# Grounding Guard
# ---------------------------------------------------------------------------

def apply_grounding_guard(candidate: Dict[str, Any], task: Task, schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nulls out nullable free-text fields whose values are not grounded in the input text.
    Removes ungrounded items from entity lists.
    Leaves enums, booleans, and numbers untouched (model judgment).
    """
    if not isinstance(candidate, dict):
        return candidate

    defs = schema.get("$defs", {})
    plan = build_schema_plan(schema, defs)
    input_lower = task.input_text.lower()

    return _ground_object(candidate, plan, input_lower)


def _ground_object(obj: Dict[str, Any], plan: SchemaPlan, input_lower: str) -> Dict[str, Any]:
    result = dict(obj)

    for fp in plan.fields:
        if fp.name not in result:
            continue
        value = result[fp.name]

        if value is None:
            continue

        # Only guard nullable free-text fields
        if fp.field_type == "string" and fp.nullable:
            if not _is_grounded(value, input_lower):
                result[fp.name] = None

        # Guard arrays of strings (entity lists) — remove ungrounded items
        elif fp.field_type == "array" and fp.items_type == "string":
            if isinstance(value, list):
                grounded = [item for item in value if _is_grounded(item, input_lower)]
                # Keep at least the original if all would be removed (trust model)
                if grounded or not value:
                    result[fp.name] = grounded

        # Recurse into nested objects
        elif fp.field_type == "object" and fp.nested_plan and isinstance(value, dict):
            result[fp.name] = _ground_object(value, fp.nested_plan, input_lower)

        # Guard arrays of objects — recurse into each
        elif fp.field_type == "array" and fp.items_plan and isinstance(value, list):
            result[fp.name] = [
                _ground_object(item, fp.items_plan, input_lower)
                if isinstance(item, dict) else item
                for item in value
            ]

    return result


def _is_grounded(value: Any, input_lower: str) -> bool:
    """Checks if a string value has evidence in the input text."""
    if not isinstance(value, str) or not value.strip():
        return True  # Empty or non-string always passes

    val_lower = value.lower().strip()

    # Direct substring match
    if val_lower in input_lower:
        return True

    # Check if significant words (>3 chars) appear in input
    words = [w for w in val_lower.split() if len(w) > 3]
    if not words:
        return True  # Short tokens — trust model

    found = sum(1 for w in words if w in input_lower)
    # At least 50% of significant words must be grounded
    return found >= max(1, len(words) * 0.5)


# ---------------------------------------------------------------------------
# Compact Prompt Builder
# ---------------------------------------------------------------------------

def _build_compact_prompt(task: Task, plan: SchemaPlan) -> str:
    """Builds a compact, schema-aware extraction prompt."""
    parts = []

    # Field-path constraints summary
    constraints = []
    for fp in plan.fields:
        c = f"- {fp.name}"
        tags = []
        if fp.nullable:
            tags.append("nullable")
        if fp.required:
            tags.append("required")
        if fp.field_type == "enum" and fp.enum_values:
            tags.append(f"enum:{fp.enum_values}")
        if fp.field_type in ("number", "integer"):
            tags.append(fp.field_type)
        if fp.field_type == "boolean":
            tags.append("bool")
        if fp.field_type == "array":
            tags.append("list")
        if tags:
            c += f" [{', '.join(tags)}]"
        if fp.description:
            c += f" — {fp.description}"
        constraints.append(c)

    parts.append("FIELD CONSTRAINTS:")
    parts.extend(constraints)
    parts.append("")
    parts.append(f"INSTRUCTION: {task.instruction}")
    parts.append("")
    parts.append(f"TEXT:\n{task.input_text}")

    return "\n".join(parts)


_SYSTEM_PROMPT = (
    "You are a precise structured-data extraction agent.\n"
    "Rules:\n"
    "1. Extract ONLY from the provided text. Never use external knowledge.\n"
    "2. Return null for nullable fields when the information is NOT explicitly in the text.\n"
    "3. For enums, use EXACTLY one of the allowed values (case-sensitive).\n"
    "4. For numbers/booleans, extract the exact value from text.\n"
    "5. For free-text fields, stay faithful to the source text.\n"
    "6. Respond in the same language as the source text."
)


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class SieveFastAgent(GenSIEAgent):
    """
    sieve-fast: One model call + deterministic repair + grounding guard.
    """

    def run(self, task: Task, model: str) -> Dict[str, Any]:
        client = _get_client()
        schema = task.target_schema
        defs = schema.get("$defs", {})
        plan = build_schema_plan(schema, defs)

        user_prompt = _build_compact_prompt(task, plan)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "extraction",
                        "schema": schema,
                        "strict": True,
                    },
                },
                temperature=0.0,
                max_tokens=4096,
            )
            content = response.choices[0].message.content
            candidate = json.loads(content)
        except Exception as e:
            logger.error(f"sieve-fast model call failed: {e}")
            return build_empty(schema, defs)

        # Repair
        repaired = repair_to_schema(candidate, schema, defs)
        # Grounding guard
        grounded = apply_grounding_guard(repaired, task, schema)
        return grounded


class SieveVerifiedAgent(GenSIEAgent):
    """
    sieve-verified: sieve-fast + adaptive verifier call for high-risk outputs.
    Only triggers verification when the output has nullable fields filled (potential hallucination).
    """

    def __init__(self):
        self._fast = SieveFastAgent()

    def _needs_verification(self, result: Dict[str, Any], plan: SchemaPlan) -> List[str]:
        """Returns list of field names that are high-risk (nullable but filled)."""
        risky = []
        for fp in plan.fields:
            if fp.nullable and fp.name in result and result[fp.name] is not None:
                risky.append(fp.name)
        return risky

    def run(self, task: Task, model: str) -> Dict[str, Any]:
        # Run fast pipeline first
        result = self._fast.run(task, model)

        schema = task.target_schema
        defs = schema.get("$defs", {})
        plan = build_schema_plan(schema, defs)

        risky_fields = self._needs_verification(result, plan)
        if not risky_fields:
            return result  # No high-risk fields, skip verification

        # Adaptive verification: ask model to confirm risky fields
        client = _get_client()
        verify_prompt = self._build_verify_prompt(task, result, risky_fields)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a verification agent. For each field listed, "
                            "determine if the value is explicitly supported by the text. "
                            "Reply with JSON: {\"field_name\": true/false} where true means "
                            "the value IS grounded in the text."
                        ),
                    },
                    {"role": "user", "content": verify_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=512,
            )
            verification = json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.warning(f"sieve-verified verification call failed: {e}")
            return result  # Fall back to fast result

        # Null out fields that verification says are ungrounded
        for field_name in risky_fields:
            if not verification.get(field_name, True):
                result[field_name] = None

        return result

    def _build_verify_prompt(self, task: Task, result: Dict[str, Any], risky_fields: List[str]) -> str:
        parts = [
            "Verify if these extracted values are EXPLICITLY present in the source text.",
            "Return true if grounded, false if hallucinated.",
            "",
            "FIELDS TO VERIFY:",
        ]
        for fname in risky_fields:
            parts.append(f'  - "{fname}": {json.dumps(result[fname], ensure_ascii=False)}')
        parts.append("")
        parts.append(f"SOURCE TEXT:\n{task.input_text}")
        return "\n".join(parts)


class SieveFewShotAgent(GenSIEAgent):
    """
    sieve-fewshot: sieve-fast + one compact schema-similar example when budget-safe.
    Local-only retrieval using schema title, domain, field overlap, and complexity.
    """

    def __init__(self, data_dirs: Optional[List[str]] = None):
        self._data_dirs = data_dirs or ["data/dev", "data/starter"]
        self._examples: Optional[List[Dict[str, Any]]] = None

    def _load_examples(self) -> List[Dict[str, Any]]:
        if self._examples is not None:
            return self._examples
        self._examples = []
        for data_dir in self._data_dirs:
            data_path = Path(data_dir)
            if not data_path.is_dir():
                continue
            for f in data_path.glob("*.json"):
                try:
                    with open(f, "r", encoding="utf-8") as fp:
                        data = json.load(fp)
                    if data.get("output") is not None:
                        self._examples.append(data)
                except Exception:
                    continue
        logger.info(f"sieve-fewshot: loaded {len(self._examples)} examples")
        return self._examples

    def _find_best_example(self, task: Task) -> Optional[Dict[str, Any]]:
        """Finds the single best matching example for the task."""
        examples = self._load_examples()
        task_fields = set(task.target_schema.get("properties", {}).keys())
        task_domain = task.metadata.get("domain", "")
        task_subdomain = task.metadata.get("subdomain", "")
        task_title = task.target_schema.get("title", "")
        task_enums = self._extract_enum_values(task.target_schema)

        best_score = -1.0
        best_ex = None

        for ex in examples:
            if ex.get("id") == task.id:
                continue

            score = 0.0
            ex_schema = ex.get("target_schema", {})
            ex_fields = set(ex_schema.get("properties", {}).keys())

            # Same schema title = very strong signal
            if ex_schema.get("title") == task_title and task_title:
                score += 10.0

            # Domain/subdomain match
            ex_meta = ex.get("metadata", {})
            if ex_meta.get("domain") == task_domain and task_domain:
                score += 2.0
            if ex_meta.get("subdomain") == task_subdomain and task_subdomain:
                score += 3.0

            # Field overlap (Jaccard)
            if task_fields or ex_fields:
                jaccard = len(task_fields & ex_fields) / len(task_fields | ex_fields) if (task_fields | ex_fields) else 0
                score += jaccard * 5.0

            # Enum overlap
            ex_enums = self._extract_enum_values(ex_schema)
            if task_enums and ex_enums:
                enum_overlap = len(task_enums & ex_enums) / len(task_enums | ex_enums)
                score += enum_overlap * 3.0

            if score > best_score:
                best_score = score
                best_ex = ex

        return best_ex if best_score > 3.0 else None

    def _extract_enum_values(self, schema: Dict[str, Any]) -> set:
        """Extracts all enum values from schema $defs."""
        values = set()
        for d in schema.get("$defs", {}).values():
            if isinstance(d, dict) and "enum" in d:
                values.update(d["enum"])
        return values

    def _format_example(self, ex: Dict[str, Any], max_text_len: int = 400) -> str:
        """Formats example compactly."""
        text = ex.get("input_text", "")
        if len(text) > max_text_len:
            text = text[:max_text_len] + "..."
        output_str = json.dumps(ex.get("output", {}), indent=2, ensure_ascii=False)
        return (
            f"--- EXAMPLE ---\n"
            f"Instruction: {ex.get('instruction', '')}\n"
            f"Text: {text}\n"
            f"Correct output:\n{output_str}\n"
            f"--- END EXAMPLE ---"
        )

    def run(self, task: Task, model: str) -> Dict[str, Any]:
        client = _get_client()
        schema = task.target_schema
        defs = schema.get("$defs", {})
        plan = build_schema_plan(schema, defs)

        # Find example
        example = self._find_best_example(task)

        # Build prompt
        user_parts = []
        if example:
            user_parts.append(self._format_example(example))
            user_parts.append("")
            user_parts.append("Now extract from the following task using the same pattern:\n")

        user_parts.append(_build_compact_prompt(task, plan))
        user_prompt = "\n".join(user_parts)

        # Budget check: if prompt is too large, skip example
        if len(user_prompt) > 28000 and example:
            user_prompt = _build_compact_prompt(task, plan)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "extraction",
                        "schema": schema,
                        "strict": True,
                    },
                },
                temperature=0.0,
                max_tokens=4096,
            )
            content = response.choices[0].message.content
            candidate = json.loads(content)
        except Exception as e:
            logger.error(f"sieve-fewshot model call failed: {e}")
            return build_empty(schema, defs)

        # Repair + grounding
        repaired = repair_to_schema(candidate, schema, defs)
        grounded = apply_grounding_guard(repaired, task, schema)
        return grounded
