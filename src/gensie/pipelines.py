"""
Custom pipelines for GenSIE competition.
Each pipeline inherits from GenSIEAgent and implements a different strategy.
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from openai import OpenAI
from gensie.agent import GenSIEAgent
from gensie.task import Task
from dotenv import load_dotenv
from logging import getLogger

load_dotenv()
logger = getLogger("gensie")


def get_client() -> OpenAI:
    return OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY", "sk-dummy"),
    )


# ---------------------------------------------------------------------------
# Schema analysis utilities
# ---------------------------------------------------------------------------

def analyze_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pre-analyzes a JSON Schema to extract useful metadata for prompt building:
    - null_trap_fields: fields that accept null (hallucination traps)
    - enum_fields: fields with enum constraints
    - rigid_fields: numbers, booleans, enums (exact match scoring)
    - free_text_fields: plain strings (semantic similarity scoring)
    - array_fields: list fields
    - nested_fields: object fields
    - required_fields: required field names
    """
    properties = schema.get("properties", {})
    defs = schema.get("$defs", {})
    required = set(schema.get("required", []))

    null_traps: List[Dict[str, str]] = []
    enum_fields: List[Dict[str, Any]] = []
    rigid_fields: List[str] = []
    free_text_fields: List[str] = []
    array_fields: List[str] = []
    nested_fields: List[str] = []

    for field_name, field_def in properties.items():
        resolved = _resolve_field(field_def, defs)

        # Detect null traps: anyOf with null type
        if _is_nullable(field_def):
            desc = field_def.get("description", resolved.get("description", ""))
            null_traps.append({"field": field_name, "description": desc})

        # Detect enums (direct or via $ref)
        if "enum" in resolved:
            enum_fields.append({
                "field": field_name,
                "values": resolved["enum"],
                "description": resolved.get("description", field_def.get("description", "")),
            })
            rigid_fields.append(field_name)
        elif resolved.get("type") in ("number", "integer", "boolean"):
            rigid_fields.append(field_name)
        elif resolved.get("type") == "string" and "enum" not in resolved:
            free_text_fields.append(field_name)
        elif resolved.get("type") == "array":
            array_fields.append(field_name)
        elif resolved.get("type") == "object":
            nested_fields.append(field_name)

    return {
        "null_traps": null_traps,
        "enum_fields": enum_fields,
        "rigid_fields": rigid_fields,
        "free_text_fields": free_text_fields,
        "array_fields": array_fields,
        "nested_fields": nested_fields,
        "required": list(required),
    }


def _resolve_field(field_def: Dict[str, Any], defs: Dict[str, Any]) -> Dict[str, Any]:
    """Resolves $ref to its definition."""
    if "$ref" in field_def:
        ref_path = field_def["$ref"]
        # e.g. "#/$defs/SoftwareType"
        parts = ref_path.replace("#/", "").split("/")
        curr = {"$defs": defs}
        for p in parts:
            curr = curr.get(p, {})
        return curr

    # anyOf case: find the non-null branch
    if "anyOf" in field_def:
        for option in field_def["anyOf"]:
            if option.get("type") != "null":
                return _resolve_field(option, defs)

    # items for arrays
    if field_def.get("type") == "array" and "items" in field_def:
        return field_def

    return field_def


def _is_nullable(field_def: Dict[str, Any]) -> bool:
    """Checks if a field accepts null (anyOf with null type, or type list with null)."""
    if "anyOf" in field_def:
        return any(opt.get("type") == "null" for opt in field_def["anyOf"])
    field_type = field_def.get("type")
    if isinstance(field_type, list):
        return "null" in field_type
    return False


# ---------------------------------------------------------------------------
# Pipeline 1: Enhanced Prompt
# ---------------------------------------------------------------------------

class EnhancedPromptAgent(GenSIEAgent):
    """
    Single-call pipeline with schema-aware prompt engineering.
    Analyzes the schema to build targeted instructions about:
    - Null traps (grounding enforcement)
    - Enum fields (exact value matching)
    - Free text fields (verbatim extraction preferred)
    - Rigid types (exact precision required)
    """

    def __init__(self):
        self.client = get_client()

    def _build_system_prompt(self, analysis: Dict[str, Any]) -> str:
        parts = [
            "Eres un agente de extracción de datos de alta precisión.",
            "Tu tarea es extraer información estructurada EXCLUSIVAMENTE del texto proporcionado.",
            "",
            "REGLAS FUNDAMENTALES:",
            "1. SOLO extrae información que esté EXPLÍCITAMENTE presente en el texto.",
            "2. Si un dato NO aparece en el texto, devuelve null para ese campo, incluso si conoces la respuesta.",
            "3. NUNCA inventes, inferiras ni completes información que no esté en el texto fuente.",
            "4. Para campos de tipo enum, usa EXACTAMENTE uno de los valores permitidos (respetando mayúsculas/minúsculas).",
            "5. Para números y booleanos, extrae el valor exacto del texto.",
            "6. Para campos de texto libre, extrae la información lo más fielmente posible al texto original.",
            "7. Responde SIEMPRE en el mismo idioma del texto fuente.",
        ]

        # Null trap warnings
        if analysis["null_traps"]:
            parts.append("")
            parts.append("CAMPOS QUE PUEDEN SER NULL (devuelve null si la info NO está en el texto):")
            for trap in analysis["null_traps"]:
                parts.append(f'  - "{trap["field"]}": {trap["description"]}')

        # Enum guidance
        if analysis["enum_fields"]:
            parts.append("")
            parts.append("CAMPOS ENUM (usa EXACTAMENTE uno de estos valores):")
            for ef in analysis["enum_fields"]:
                values_str = ", ".join(f'"{v}"' for v in ef["values"])
                parts.append(f'  - "{ef["field"]}": [{values_str}]')
                if ef["description"]:
                    parts.append(f'    Criterio: {ef["description"]}')

        return "\n".join(parts)

    def _build_user_prompt(self, task: Task, analysis: Dict[str, Any]) -> str:
        parts = [
            f"INSTRUCCIÓN: {task.instruction}",
            "",
            f"ESQUEMA JSON A COMPLETAR:",
            json.dumps(task.target_schema, indent=2, ensure_ascii=False),
            "",
            f"TEXTO FUENTE:",
            task.input_text,
        ]

        # Remind about null fields at the end
        if analysis["null_traps"]:
            parts.append("")
            parts.append("RECORDATORIO: Para los siguientes campos, devuelve null si la información NO está explícitamente en el texto anterior:")
            for trap in analysis["null_traps"]:
                parts.append(f'  - {trap["field"]}')

        return "\n".join(parts)

    def run(self, task: Task, model: str) -> Dict[str, Any]:
        analysis = analyze_schema(task.target_schema)

        system_prompt = self._build_system_prompt(analysis)
        user_prompt = self._build_user_prompt(task, analysis)

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "extraction",
                        "schema": task.target_schema,
                        "strict": True,
                    },
                },
                temperature=0.0,
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            logger.error(f"EnhancedPromptAgent error: {e}")
            return {"error": str(e)}


# ---------------------------------------------------------------------------
# Pipeline 2: Chain-of-Thought + Extract (two-step)
# ---------------------------------------------------------------------------

class CoTExtractAgent(GenSIEAgent):
    """
    Two-step pipeline:
    1. Reasoning step: Ask the model to analyze the text and schema in free text,
       identifying what can be extracted and what should be null.
    2. Extraction step: Use the reasoning as context for constrained JSON extraction.
    """

    def __init__(self):
        self.client = get_client()

    def _build_reasoning_prompt(self, task: Task, analysis: Dict[str, Any]) -> str:
        null_fields_list = ""
        if analysis["null_traps"]:
            items = [f'  - "{t["field"]}": {t["description"]}' for t in analysis["null_traps"]]
            null_fields_list = "\nCAMPOS QUE PUEDEN SER NULL:\n" + "\n".join(items)

        enum_list = ""
        if analysis["enum_fields"]:
            items = []
            for ef in analysis["enum_fields"]:
                values_str = ", ".join(f'"{v}"' for v in ef["values"])
                items.append(f'  - "{ef["field"]}": [{values_str}] — {ef["description"]}')
            enum_list = "\nCAMPOS ENUM:\n" + "\n".join(items)

        return (
            f"Analiza el siguiente texto y esquema de extracción. "
            f"Para CADA campo del esquema, indica:\n"
            f"- Si la información está explícitamente en el texto y cuál es el valor.\n"
            f"- Si la información NO está en el texto y por tanto debe ser null.\n"
            f"- Para campos enum, qué valor corresponde y por qué.\n"
            f"- Para campos numéricos, el valor exacto encontrado.\n"
            f"\nSé breve y directo. Un análisis por campo.\n"
            f"{null_fields_list}"
            f"{enum_list}"
            f"\nINSTRUCCIÓN DE LA TAREA: {task.instruction}\n"
            f"\nESQUEMA:\n{json.dumps(task.target_schema, indent=2, ensure_ascii=False)}\n"
            f"\nTEXTO:\n{task.input_text}"
        )

    def _build_extraction_prompt(self, task: Task, reasoning: str) -> str:
        return (
            f"Basándote en el siguiente análisis previo, extrae los datos estructurados.\n\n"
            f"ANÁLISIS:\n{reasoning}\n\n"
            f"INSTRUCCIÓN: {task.instruction}\n\n"
            f"ESQUEMA:\n{json.dumps(task.target_schema, indent=2, ensure_ascii=False)}\n\n"
            f"TEXTO ORIGINAL:\n{task.input_text}\n\n"
            f"Genera el JSON de extracción. Usa null para cualquier campo cuya información "
            f"no esté explícitamente en el texto."
        )

    def run(self, task: Task, model: str) -> Dict[str, Any]:
        analysis = analyze_schema(task.target_schema)

        # Step 1: Reasoning (free text, no constrained decoding)
        try:
            reasoning_prompt = self._build_reasoning_prompt(task, analysis)

            reasoning_response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Eres un analista de textos. Tu trabajo es examinar un texto "
                            "y determinar qué información puede extraerse según un esquema dado. "
                            "Sé conciso: máximo 2-3 líneas por campo."
                        ),
                    },
                    {"role": "user", "content": reasoning_prompt},
                ],
                temperature=0.0,
                max_tokens=1500,  # Keep reasoning short to save budget
            )

            reasoning = reasoning_response.choices[0].message.content or ""
            logger.info(f"CoT reasoning length: {len(reasoning)} chars")

        except Exception as e:
            logger.warning(f"CoT reasoning failed, falling back to direct extraction: {e}")
            reasoning = ""

        # Step 2: Extraction (constrained decoding with reasoning as context)
        try:
            extraction_prompt = self._build_extraction_prompt(task, reasoning)

            extraction_response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Eres un agente de extracción preciso. Extrae SOLO información "
                            "presente en el texto. Devuelve null para datos ausentes."
                        ),
                    },
                    {"role": "user", "content": extraction_prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "extraction",
                        "schema": task.target_schema,
                        "strict": True,
                    },
                },
                temperature=0.0,
            )

            content = extraction_response.choices[0].message.content
            return json.loads(content)

        except Exception as e:
            logger.error(f"CoTExtractAgent extraction error: {e}")
            return {"error": str(e)}


# ---------------------------------------------------------------------------
# Pipeline 3: Self-Correction
# ---------------------------------------------------------------------------

class SelfCorrectAgent(GenSIEAgent):
    """
    Two-step pipeline with validation:
    1. Extract with enhanced prompt (reuses EnhancedPromptAgent logic).
    2. Validate the output for common issues (null-trap violations, empty arrays).
    3. If issues found, send a targeted correction prompt with specific feedback.
    """

    def __init__(self):
        self.client = get_client()

    def _extract(self, task: Task, model: str, analysis: Dict[str, Any],
                 correction_context: Optional[str] = None) -> Dict[str, Any]:
        """Runs one extraction call, optionally with correction context."""
        system_parts = [
            "Eres un agente de extracción de datos de alta precisión.",
            "Extrae información SOLO del texto proporcionado.",
            "Devuelve null para cualquier dato que NO esté explícitamente en el texto,",
            "incluso si conoces la respuesta por conocimiento general.",
        ]

        if analysis["enum_fields"]:
            system_parts.append("")
            system_parts.append("CAMPOS ENUM (valores exactos obligatorios):")
            for ef in analysis["enum_fields"]:
                values_str = ", ".join(f'"{v}"' for v in ef["values"])
                system_parts.append(f'  - "{ef["field"]}": [{values_str}]')

        user_parts = []

        if correction_context:
            user_parts.append(f"CORRECCIÓN REQUERIDA:\n{correction_context}\n")

        user_parts.extend([
            f"INSTRUCCIÓN: {task.instruction}",
            "",
            f"ESQUEMA:\n{json.dumps(task.target_schema, indent=2, ensure_ascii=False)}",
            "",
            f"TEXTO:\n{task.input_text}",
        ])

        if analysis["null_traps"]:
            user_parts.append("")
            user_parts.append("RECORDATORIO NULL: Devuelve null para estos campos si la info no está en el texto:")
            for trap in analysis["null_traps"]:
                user_parts.append(f'  - {trap["field"]}: {trap["description"]}')

        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "\n".join(system_parts)},
                {"role": "user", "content": "\n".join(user_parts)},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "extraction",
                    "schema": task.target_schema,
                    "strict": True,
                },
            },
            temperature=0.0,
        )
        content = response.choices[0].message.content
        return json.loads(content)

    def _validate(self, result: Dict[str, Any], task: Task,
                  analysis: Dict[str, Any]) -> Optional[str]:
        """
        Validates the extraction result. Returns correction feedback if issues
        are found, None if the result looks good.
        """
        issues = []

        # Check null traps: if a nullable field has a value, flag it for review
        # We can't know for sure it's wrong, but we can ask the model to double-check
        null_trap_names = {t["field"] for t in analysis["null_traps"]}
        non_null_traps = []
        for field_name in null_trap_names:
            if field_name in result and result[field_name] is not None:
                desc = next(
                    (t["description"] for t in analysis["null_traps"]
                     if t["field"] == field_name), ""
                )
                non_null_traps.append(f'  - "{field_name}" = {json.dumps(result[field_name], ensure_ascii=False)}')
                non_null_traps.append(f'    Descripción: {desc}')

        if non_null_traps:
            issues.append(
                "Los siguientes campos nullable tienen valores asignados. "
                "Verifica que esta información aparece LITERALMENTE en el texto. "
                "Si NO está en el texto, cámbialo a null:\n" + "\n".join(non_null_traps)
            )

        # Check enum fields: verify value is valid
        for ef in analysis["enum_fields"]:
            field_name = ef["field"]
            if field_name in result and result[field_name] not in ef["values"]:
                issues.append(
                    f'Campo enum "{field_name}" tiene valor "{result[field_name]}" '
                    f'que no está en los valores permitidos: {ef["values"]}'
                )

        # Check for suspiciously empty required arrays
        for field_name in analysis["array_fields"]:
            if field_name in result and isinstance(result[field_name], list):
                if len(result[field_name]) == 0 and field_name in analysis["required"]:
                    issues.append(
                        f'Campo requerido "{field_name}" es un array vacío. '
                        f"Revisa el texto para extraer elementos si los hay."
                    )

        if not issues:
            return None

        return "Tu extracción anterior tiene posibles problemas:\n\n" + "\n\n".join(issues) + \
            "\n\nExtracción anterior:\n" + json.dumps(result, indent=2, ensure_ascii=False)

    def run(self, task: Task, model: str) -> Dict[str, Any]:
        analysis = analyze_schema(task.target_schema)

        # Step 1: Initial extraction
        try:
            result = self._extract(task, model, analysis)
        except Exception as e:
            logger.error(f"SelfCorrectAgent initial extraction error: {e}")
            return {"error": str(e)}

        # Step 2: Validate
        correction = self._validate(result, task, analysis)

        if correction is None:
            return result  # Looks good, no correction needed

        # Step 3: Correction pass
        logger.info("SelfCorrectAgent: issues detected, running correction pass")
        try:
            corrected = self._extract(task, model, analysis, correction_context=correction)
            return corrected
        except Exception as e:
            logger.warning(f"SelfCorrectAgent correction failed, returning original: {e}")
            return result  # Fall back to original if correction fails


# ---------------------------------------------------------------------------
# Pipeline 4: Few-Shot RAG
# ---------------------------------------------------------------------------

def _schema_field_names(schema: Dict[str, Any]) -> set:
    """Extracts top-level property names from a JSON Schema."""
    return set(schema.get("properties", {}).keys())


def _schema_similarity(schema_a: Dict[str, Any], schema_b: Dict[str, Any]) -> float:
    """Jaccard similarity between two schemas based on field names."""
    fields_a = _schema_field_names(schema_a)
    fields_b = _schema_field_names(schema_b)
    if not fields_a and not fields_b:
        return 1.0
    if not fields_a or not fields_b:
        return 0.0
    intersection = fields_a & fields_b
    union = fields_a | fields_b
    return len(intersection) / len(union)


class FewShotAgent(GenSIEAgent):
    """
    RAG-based pipeline that finds similar examples from the dev set and
    includes them as few-shot demonstrations.

    Selection criteria (in priority order):
    1. Same domain/subdomain (metadata match)
    2. Schema structure similarity (Jaccard of field names)

    Includes 1-2 examples to stay within token budget.
    """

    def __init__(self, data_dirs: Optional[List[str]] = None):
        self.client = get_client()
        self._examples: List[Dict[str, Any]] = []
        self._loaded = False
        self._data_dirs = data_dirs or ["data/dev", "data/starter"]

    def _load_examples(self):
        """Lazily loads all available examples from data directories."""
        if self._loaded:
            return

        for data_dir in self._data_dirs:
            data_path = Path(data_dir)
            if not data_path.is_dir():
                continue
            for f in data_path.glob("*.json"):
                try:
                    with open(f, "r", encoding="utf-8") as fp:
                        data = json.load(fp)
                    # Only keep examples that have output (gold/silver data)
                    if data.get("output") is not None:
                        self._examples.append(data)
                except Exception:
                    continue

        self._loaded = True
        logger.info(f"FewShotAgent: loaded {len(self._examples)} examples")

    def _find_similar(self, task: Task, max_examples: int = 2) -> List[Dict[str, Any]]:
        """Finds the most similar examples to the given task."""
        self._load_examples()

        task_domain = task.metadata.get("domain", "")
        task_subdomain = task.metadata.get("subdomain", "")

        scored = []
        for ex in self._examples:
            # Don't use the same task as its own example
            if ex.get("id") == task.id:
                continue

            score = 0.0

            # Domain match bonus
            ex_domain = ex.get("metadata", {}).get("domain", "")
            ex_subdomain = ex.get("metadata", {}).get("subdomain", "")

            if ex_domain == task_domain and task_domain:
                score += 2.0
            if ex_subdomain == task_subdomain and task_subdomain:
                score += 3.0

            # Schema similarity
            schema_sim = _schema_similarity(task.target_schema, ex.get("target_schema", {}))
            score += schema_sim * 5.0  # Schema match is heavily weighted

            scored.append((score, ex))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ex for _, ex in scored[:max_examples]]

    def _format_example(self, ex: Dict[str, Any]) -> str:
        """Formats a single example for inclusion in the prompt."""
        # Truncate input_text to save tokens
        text = ex.get("input_text", "")
        if len(text) > 500:
            text = text[:500] + "..."

        return (
            f"--- EJEMPLO ---\n"
            f"Instrucción: {ex.get('instruction', '')}\n"
            f"Esquema: {json.dumps(ex.get('target_schema', {}), ensure_ascii=False)}\n"
            f"Texto: {text}\n"
            f"Resultado correcto:\n{json.dumps(ex.get('output', {}), indent=2, ensure_ascii=False)}\n"
            f"--- FIN EJEMPLO ---"
        )

    def run(self, task: Task, model: str) -> Dict[str, Any]:
        analysis = analyze_schema(task.target_schema)
        similar = self._find_similar(task, max_examples=2)

        # Build system prompt
        system_parts = [
            "Eres un agente de extracción de datos de alta precisión.",
            "Extrae información SOLO del texto proporcionado.",
            "Devuelve null para cualquier dato que NO esté explícitamente en el texto.",
        ]

        if analysis["null_traps"]:
            system_parts.append("")
            system_parts.append("CAMPOS NULLABLE (null si la info no está en el texto):")
            for trap in analysis["null_traps"]:
                system_parts.append(f'  - "{trap["field"]}": {trap["description"]}')

        if analysis["enum_fields"]:
            system_parts.append("")
            system_parts.append("CAMPOS ENUM (valores exactos):")
            for ef in analysis["enum_fields"]:
                values_str = ", ".join(f'"{v}"' for v in ef["values"])
                system_parts.append(f'  - "{ef["field"]}": [{values_str}]')

        # Build user prompt with examples
        user_parts = []

        if similar:
            user_parts.append("Aquí tienes ejemplos de extracciones correctas similares:\n")
            for ex in similar:
                user_parts.append(self._format_example(ex))
            user_parts.append("")
            user_parts.append("Ahora extrae los datos de la siguiente tarea, siguiendo el mismo patrón:\n")

        user_parts.extend([
            f"INSTRUCCIÓN: {task.instruction}",
            "",
            f"ESQUEMA:\n{json.dumps(task.target_schema, indent=2, ensure_ascii=False)}",
            "",
            f"TEXTO:\n{task.input_text}",
        ])

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "\n".join(system_parts)},
                    {"role": "user", "content": "\n".join(user_parts)},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "extraction",
                        "schema": task.target_schema,
                        "strict": True,
                    },
                },
                temperature=0.0,
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            logger.error(f"FewShotAgent error: {e}")
            return {"error": str(e)}
