import os
import json
from typing import Any, Dict
from openai import OpenAI
from gensie.agent import GenSIEAgent, Participant, ParticipantInfo, PipelineInfo
from gensie.task import Task
from gensie.pipelines import EnhancedPromptAgent, CoTExtractAgent, SelfCorrectAgent, FewShotAgent
from dotenv import load_dotenv
from logging import getLogger

load_dotenv()
logger = getLogger("gensie")


class BasicAgent(GenSIEAgent):
    """
    Reference implementation using OpenAI Structured Outputs.
    Configurable via environment variables:
    - OPENAI_BASE_URL: (Optional) Custom endpoint for local LLMs.
    - OPENAI_API_KEY: (Required) Your API key.
    """

    def __init__(self):
        self.client = OpenAI(
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY", "sk-dummy"),
        )

    def run(self, task: Task, model: str) -> Dict[str, Any]:
        """
        Executes the extraction using OpenAI's response_format for strict schema compliance.
        """
        prompt = task.get_input_prompt()

        # Call OpenAI with the task's JSON schema
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise data extraction agent.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "extraction",
                    "schema": task.target_schema,
                    "strict": True,
                },
            },
        )

        # Parse the structured JSON response
        try:
            content = response.choices[0].message.content
            return json.loads(content)
        except (json.JSONDecodeError, AttributeError, IndexError) as e:
            # Fallback for unexpected API errors
            return {"error": f"Failed to parse model response: {str(e)}"}
        except Exception as e:
            logger.error(str(e))
            return {"error": str(e)}


class OfficialParticipant(Participant):
    """
    Standard entry point for the competition.
    Participants can configure up to 3 pipelines here.
    """

    def __init__(self):
        self.pipelines = {
            "baseline": BasicAgent(),
            "enhanced-prompt": EnhancedPromptAgent(),
            "cot-extract": CoTExtractAgent(),
            "self-correct": SelfCorrectAgent(),
            "few-shot": FewShotAgent(),
        }

    def get_info(self) -> ParticipantInfo:
        return ParticipantInfo(
            team_name="GenSIE Baseline Team",
            institution="Official",
            pipelines=[
                PipelineInfo(
                    name="baseline",
                    description="Standard OpenAI agent using structured outputs.",
                ),
                PipelineInfo(
                    name="enhanced-prompt",
                    description="Schema-aware prompt engineering with null-trap detection.",
                ),
                PipelineInfo(
                    name="cot-extract",
                    description="Two-step Chain-of-Thought reasoning then constrained extraction.",
                ),
                PipelineInfo(
                    name="self-correct",
                    description="Extract-validate-correct loop with null-trap verification.",
                ),
                PipelineInfo(
                    name="few-shot",
                    description="RAG pipeline with schema-similar examples as demonstrations.",
                ),
            ],
        )

    def get_agent(self, pipeline_name: str) -> GenSIEAgent:
        if pipeline_name not in self.pipelines:
            # Fallback to default if pipeline not found, or raise error
            return self.pipelines["baseline"]
        return self.pipelines[pipeline_name]
