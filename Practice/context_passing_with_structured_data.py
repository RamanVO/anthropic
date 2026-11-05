import asyncio
from dotenv import load_dotenv
import json, re
from claude_agent_sdk import query, ClaudeAgentOptions, AgentDefinition, ResultMessage, AssistantMessage, ToolUseBlock,UserMessage, ToolResultBlock


load_dotenv()


from typing import Literal, Optional, TypedDict

class Finding(TypedDict):
    claim:str
    source_url:Optional[str]
    document_name:str
    page_number: Optional[int]
    confidence:Literal["high", "medium", "low"]
    retrieved_by: str 

FINDING_FORMAT=(
    'Return JSON only: {"findings": [{"claim": "...", "source_url": "... or null", '
    '"document_name": "...", "page_number": 1 or null, '
    '"confidence": "high|medium|low", "retrieved_by": "<your agent name>"}]} '
)

SYNTHESIS_FORMAT = (
    'Return JSON only: {"title": "...", "sections": [{"heading": "...", "claims": ['
    '{"text": "...", "confidence": "high|medium|low", '
    '"sources": [{"source_url": "... or null", "document_name": "...", "page_number": 1 or null}]}'
    ']}]}'
)


web_search_agent = AgentDefinition(
    description="Search the web for information relevant to the research topic.",
    prompt=(
        "You are a web search agent. Your task is to find relevant information on the web based on the subtopic provided. "
        "You will receive a structured input containing the subtopic and the overall research goal. "
        f"Your output should {FINDING_FORMAT}. Set retrieved_by to \"web_search\". Return one factual claim per finding. "
        "Ensure that the information you provide is accurate and relevant to the subtopic. "
        "Do not include any personal opinions or unrelated information. "
    ),
    tools=["WebSearch", "WebFetch"]
)

document_analysis_agent = AgentDefinition(
    description="Analyse local documents provided by the user and extract individual factual claims with page references.",
    prompt=(
        "You are a document analysis agent. You will receive the subtopic, the overall research goal "
        "and the path to a folder of documents. Use Glob to list the files in that folder, then Read each relevant one. "
        "Record the file name as document_name and the page each claim appears on as page_number. "
        f"Your output should {FINDING_FORMAT}. Set retrieved_by to \"doc_analysis\". Return one factual claim per finding. "
        "Ensure that the information you provide is accurate and relevant to the subtopic. "
        "Do not include any personal opinions or unrelated information. "
    ),
    tools=["Glob","Read"]
)

synthesis_agent = AgentDefinition(
    description=("Combines findings from web_search and doc_analysis into structured, source-attributed claims in JSON. "
                 "Use only after both research agents have returned."),
    prompt=(
            # "You are a synthesis agent. Your task is to synthesize the findings from various subagents into a comprehensive report. "
            # "You will receive structured inputs containing the subtopic, the overall research goal, and the two lists of findings one from web search and other from document analysis agents "  
            # "where each finding has: claim, source_url, document_name, page_number, confidence, retrieved_by. "
            # "Your output should be a well-organized report that integrates the information provided by the subagents, ensuring that all claims are properly cited with their source URLs and page numbers. "
            # "Ensure that the information you provide is accurate and relevant to the research goal. "
            # "Do not include any personal opinions or unrelated information. "
            # "Every claim must have at least one entry in sources, copied exactly from the findings you received. Never invent a source. "
            # "If a statement cannot be supported by a finding, leave it out."
            # f" {SYNTHESIS_FORMAT}"
            "You are a synthesis agent. You combine research findings into a structured set of attributed claims. "
            "You will receive two lists of findings, each with claim, source_url, document_name, page_number, confidence and retrieved_by. "
            "Your output must follow this exact format and contain no other keys: "
            f"{SYNTHESIS_FORMAT} "
            "Every claim must have at least one entry in sources, copied exactly from the findings you received. "
            "Never invent a source. If a statement cannot be supported by a finding, leave it out. "
        ),
     tools=[],
     model="sonnet"
)

def is_valid_source(source: dict) -> bool:
    """A source is valid if it can be traced: a URL for web pages, or a document name for local files."""
    return bool(source.get("source_url")) or bool(source.get("document_name"))

def extract_json(text: str | None) -> dict:
    """Pull a JSON object out of the model's final text, tolerating code fences or extra prose."""
    if not text:
        raise ValueError("No final output received")
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON object found in final output")
        candidate = text[start:end + 1]
    return json.loads(candidate)

def verify_attribution(report: dict) -> list[dict]:
    """Return every claim lacking a valid source. An empty list means fully attributed."""
    sections = report.get("sections", [])
    if not sections or not any("claims" in s for s in sections):
        return [{"section": "-", "claim": "-",
                 "reason": f"report does not match expected schema (top-level keys: {list(report)})"}]
  
    flagged = []
    total_claims = 0
    for section in report.get("sections", []):
        for claim in section.get("claims", []):
            total_claims += 1
            sources = claim.get("sources") or []
            if not any(isinstance(s, dict) and is_valid_source(s) for s in sources):
                flagged.append({
                    "section": section.get("heading", "?"),
                    "claim": claim.get("text", ""),
                    "reason": "no sources" if not sources else "no valid source",
                })
    if total_claims == 0:
        flagged.append({"section": "-", "claim": "-", "reason": "report contains no claims"})
    return flagged 

topic="Use of Microsoft Technology stack"
DOC_PATH=f"C:/dev/learn/anthropic/docs"
docs_dir=DOC_PATH


async def main():
    # Example usage of the query function with structured data
    final_text=None
    result_count=0
    spawn_batches = []   # one entry per coordinator response that spawned subagents
    turn = 0
    spawns_by_turn: dict[int, list[str]] = {}

    async for message in query(
        prompt=(
            f"Research {topic} and produce a set of fully attributed claims in the synthesis agent's JSON format. "
            f" Use web sources and the local documents in {docs_dir}."
            f" When you invoke the doc_analysis agent, include this folder path in its prompt."
        ),
        options=ClaudeAgentOptions(
            system_prompt=("You are the coordinator. All communication goes through you; subagents never communicate with each other. "
                "Subagents know nothing except what is in their prompt. Every prompt you write must include the subtopic, the overall research goal and complete earlier findings. "
                "Each subagent defines its own output format in its instructions; "
                f"When you invoke the synthesis agent, paste each research agent's findings JSON exactly as it was returned, "
                "as raw JSON inside the prompt. Do not convert findings to bullet points, prose or any other format. "
                f"When you invoke the synthesis agent, include this output format verbatim and add no other structure: {SYNTHESIS_FORMAT} "
                "Quality criteria: the final report is complete only when every claim is attributed to at least one source "
                "with all its available metadata. Pass all findings to the synthesis agent verbatim with their metadata intact; "
                "never summarise or drop fields. Claims with low confidence must be flagged as such in the report.\n"
                "Whenever you call the Agent tool, set run_in_background to false, so that you receive each subagent's result "
                "directly before your turn ends. Invoke web_search and doc_analysis in the same response, then invoke synthesis "
                "once both have returned. "
                "Your final answer must be the synthesis agent's JSON output, returned verbatim, with no other text.\n"
                ),
            allowed_tools=["Agent", "WebSearch", "WebFetch", "Glob", "Read"],
            add_dirs=[DOC_PATH],
            model = "claude-haiku-4-5",
            max_buffer_size=10 * 1024 * 1024,  # 10 MB
            agents={
                "web_search": web_search_agent,
                "doc_analysis": document_analysis_agent,
                "synthesis": synthesis_agent,
            },
        ),
    ):
        is_coordinator = getattr(message, "parent_tool_use_id", None) is None

        # Tool results returning to the coordinator mark the end of a turn
        if isinstance(message, UserMessage) and is_coordinator and isinstance(message.content, list):
            if any(isinstance(b, ToolResultBlock) for b in message.content):
                turn += 1

        if isinstance(message, AssistantMessage) and is_coordinator:
            agent_calls = [b for b in message.content
                        if isinstance(b, ToolUseBlock) and b.name in ("Agent", "Task")]
            for block in agent_calls:
                spawns_by_turn.setdefault(turn, []).append(block.input.get("subagent_type"))
                print(f"→ turn {turn} spawned: {block.input.get('subagent_type')}")
                # print(f"→ response {len(spawn_batches)} spawned: {batch}")
                for block in agent_calls:
                    if block.input.get("subagent_type") == "synthesis":
                        print("--- synthesis task prompt ---")
                        print(block.input.get("prompt", "")[:2000])

        if isinstance(message, ResultMessage):
            result_count += 1
            final_text = message.result

    print(f"ResultMessages received: {result_count}")
    spawn_batches = list(spawns_by_turn.values())
    check_parallel(spawn_batches)
    
    try:
        report = extract_json(final_text)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"Verification failed: could not parse report ({e})")
        print("--- Final output was ---")
        print(final_text)
        return

    flagged = verify_attribution(report)
    #if any(f["reason"] == "report contains no claims" for f in flagged):
    if any(f["reason"] == "report contains no claims" or f["reason"].startswith("report does not match") for f in flagged):
        print("--- Final output was ---")
        print(final_text)
    if flagged:
        print(f"{len(flagged)} unattributed claim(s):")
        for item in flagged:
            print(f"  [{item['section']}] {item['claim']!r}: {item['reason']}")
    else:
        print("All claims attributed.")


def check_parallel(batches: list[list[str]]) -> None:
    research = {"web_search", "doc_analysis"}
    parallel_batch = next((i for i, b in enumerate(batches) if research <= set(b)), None)
    synthesis_batch = next((i for i, b in enumerate(batches) if "synthesis" in b), None)

    if parallel_batch is None:
        print("✗ Research agents were NOT spawned in the same response")
    else:
        print(f"✓ Research agents spawned together in response {parallel_batch + 1}")

    if synthesis_batch is not None and parallel_batch is not None and synthesis_batch <= parallel_batch:
        print("✗ Synthesis was spawned before or alongside the research agents")
    elif synthesis_batch is not None:
        print(f"✓ Synthesis spawned afterwards, in response {synthesis_batch + 1}")



asyncio.run(main())