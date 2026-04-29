"""
Custom pipelines for GenSIE competition.
Each pipeline inherits from GenSIEAgent and implements a different strategy.
"""

import os
import json
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
