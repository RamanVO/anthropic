from dotenv import load_dotenv
from anthropic import Anthropic
import json

load_dotenv()
# client = Anthropic()
# model = "claude-haiku-4-5"

def web_search(query):
    # Mock search: echoes the query so the loop can be tested on any topic.
    # Replace with a real search API, or use the server-side web search tool.
    results = [
        {"title": f"Overview: {query}",
         "snippet": f"Mock result for '{query}'. This stub returns no real information."},
        {"title": f"Recent developments: {query}",
         "snippet": f"Second mock result for '{query}'."},
    ]
    return json.dumps(results)

web_search_tool_schema = {
    "name": "web_search", 
    "description": (
        "Searches the web for information and returns relevant results each with a title and a short snippet. "
        "Use this tool to find up-to-date information, news, or specific details that are not available in the model's training data."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "A search query string to find relevant information on the web. "
                    "Examples: 'latest news on climate change', 'Python programming tutorials', 'current stock prices for AAPL'."
                ),
                "minLength": 1
            }
        },
        "required": ["query"]
    }
}

TOOL_HANDLERS = {"web_search": web_search}

class HubSpokeResearchCoordinator:
    SYSTEM_PROMPT = (
        "You are the coordinator (hub) of a hub-and-spoke research system. "
        "All communication goes through you; subagents never communicate with each other. "
        "Your responsibilities:\n"
        "1. Breadth: decompose the topic into distinct, non-overlapping subtopics that together "
        "cover its full breadth. Before delegating, check that no major category is missing.\n"
        "2. Isolation: subagents know nothing except what is in their prompt. Every prompt you write "
        "must include the subtopic, the overall research goal, the expected output format and any "
        "relevant earlier findings.\n"
        "3. Gaps: after results come back, check each subtopic for substantive coverage and "
        "re-delegate targeted requests for any gaps."
    )
    SUBAGENTS = {
        "web_search_agent": {
            "description": "Finds current information, news and sources on a subtopic.",
            "system_prompt": (
                "You are a web search agent. Research the subtopic you are given. "
                'Return JSON only: {"results": [{"title": "...", "snippet": "..."}]}'
            ),
            "tools": [web_search_tool_schema],
        },
        "document_analysis_agent": {
            "description": "Analyses material to extract key points and summaries on a subtopic.",
            "system_prompt": (
                "You are a document analysis agent. Analyse the material you are given. "
                'Return JSON only: {"analysis": [{"key_point": "...", "summary": "..."}]}'
            ),
            "tools": [],   # analyses what the coordinator passes it; no searching of its own
        }
    }

    def __init__(self, model="claude-haiku-4-5"):
        self.client = Anthropic()
        self.model = model

    COVERAGE_THRESHOLD = 0.9  # 90% coverage is considered sufficient
    MAX_REFINEMENTS = 3  # Limit the number of refinement iterations to avoid infinite loops
    
    def research(self, topic, required_areas=None):
        subtopics = self.decompose(topic, required_areas)
        subtopics=subtopics[:2]
        results = self.delegate(topic, subtopics)
        coverage = self.evaluate_coverage(topic, subtopics, results)

        # Iterative refinement: re-delegate targeted requests for the gaps
        rounds = 0
        while coverage["completeness"] < self.COVERAGE_THRESHOLD:
            if rounds >= self.MAX_REFINEMENTS:
                print(f"WARNING: refinement cap reached; gaps remain: {coverage['gaps']}")
                break
            rounds += 1
            print(f"Refinement round {rounds}: {coverage['completeness']:.0%} covered, "
                  f"re-delegating {coverage['gaps']}")

            for gap in coverage["gaps"]:
                reason = coverage["coverage"][gap]["reason"]
                self._research_subtopic(topic, gap, results, gap_reason=reason)

            coverage = self.evaluate_coverage(topic, subtopics, results)

        return self.aggregate(topic, results, coverage)

    def _ask_coordinator(self, prompt):
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )
        content = "\n".join([block.text for block in response.content if block.type == "text"])
        # print(f"Coordinator response: {content}")
        return self._parse_json(content)

    @staticmethod
    def _parse_json(text):
        # Strip a Markdown code fence if Claude adds one, then parse
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]        # drop the opening ```json line
            text = text.rsplit("```", 1)[0]     # drop the closing ```
        return json.loads(text)


    
    def decompose(self, topic, required_areas=None):
        # Phase 1: Use the model to decompose the topic into subtopics
        first_prompt = (
            f"Decompose the following research topic into at least 5 distinct, non-overlapping subtopics:\n"
            f"Topic: {topic}\n"
            f"Once you have identified the subtopics, review the subtopics silently again and ensure no area is missing.\n"
            f"A subtopic must cover exactly one type. Combine two types with 'and' or '/'\n"
            f"Update the list of subtopics if necessary. Return the final list of subtopics with only the JSON array, with no other text.\n"
        )
        first_list= self._ask_coordinator(first_prompt)
        # print(f"Initial subtopics: {first_list}")

        if required_areas:
            required_line= f"User required these areas to be covered as its own subtopics: {json.dumps(required_areas)}\n\n"
        else:
            required_line=""
            
        # Phase 2: check the list for breadth, adding any missing categories
        review_prompt = (
            f"Research topic: {topic}\n"
            f"Proposed subtopics: {json.dumps(first_list)}\n\n"
            f"{required_line}"
            "Check this list for breadth in three stages.\n\n"
            "1. Choose the main dimension. If the topic is about a set of distinct types "
            "(for example technologies, methods or species), the main categories must be those "
            "types, with each type as its own category. Cross-cutting aspects such as economics, "
            "policy or environmental impact come second.\n\n"
            "2. Enumerate before checking. Your checklist must include every user-required area, even if you would consider it borderline or outside the topic."
            "List every member you would expect within the topic, "
            "including established, emerging and experimental or not-yet-commercial ones. "
            "Put this full list in \"checklist\".\n\n"
            "3. Check and fix. Each user-required area must be represented as its own subtopic. "
            "Compare every checklist item with the proposed subtopics. Add a "
            "subtopic for any item that is missing. Split any subtopic that bundles several types "
            "into one subtopic per type. Keep subtopics distinct and non-overlapping, with at "
            "least 5 and no more than 12 in total.\n\n"
            "Reply with ONLY a JSON object in this form, with no other text: "
            '{"checklist": ["..."], "categories": ["..."], "subtopics": ["..."]}'
        )

        review_response=self._ask_coordinator(review_prompt)
        # print(f"Checklist: {review_response['checklist']}")
        # print(f"Categories: {review_response['categories']}")
        return review_response["subtopics"]

    def delegate(self, topic, subtopics):
        results = {}
        for subtopic in subtopics:
            self._research_subtopic(topic, subtopic, results)
        return results

    def _research_subtopic(self, topic, subtopic, results, gap_reason=None):
        search_store=results.setdefault("web_search_agent", {})
        analysis_store=results.setdefault("document_analysis_agent", {})
        prior=search_store.get(subtopic, {}).get("results", [])


        # Search: on a refinement pass, tell the agent what's missing and what it already has

        search_prompt= (
            f"Subtopic: {subtopic}\n"
            f"Overall research goal: {topic}\n"
        )

        if gap_reason:
            search_prompt+= (
                f"Previous search results were insufficient because: {gap_reason}\n"
                f"Results already found (do no repeat these ): {json.dumps(prior)}\n"
                "Please focus on finding additional sources that address this gap.\n"
            )

        search_prompt+= (
            'Return JSON only: {"results": [{"title": "...", "snippet": "..."}]}\n'
        )

        new = self._call_subagent("web_search_agent", search_prompt).get("results", [])
        combined = prior + new
        search_store[subtopic] = {"results": combined}

        # Analysis: re-run over ALL results, so the analysis reflects the full picture

        analysis_prompt= (
            f"for the Subtopic: {subtopic} with an overall research goal of {topic}, the web search results are:{json.dumps(combined)}  \n"
            f"Conduct thorough analysis and synthesis, identifying key findings, recurring themes, patterns, contradictions, evidence gaps, and actionable insights. \n"
            f"Evaluate source credibility, relevance, and evidence quality. Preserve critical details while eliminating redundancy. Do not fabricate information or conclusions. \n"
            f"Cite sources for significant claims and clearly distinguish facts from interpretations.\n"
            f"Return the results for this subtopic in valid JSON format only, adhering to the specified schema\n"
            '{"analysis": [{"key_point": "...", "summary": "..."}]}\n'
        )

        analysis_store[subtopic] = self._call_subagent("document_analysis_agent", analysis_prompt)


    def aggregate(self, topic, results, coverage):
        # Combine results from sub-agents into a coherent summary
        print(f"Final Coverage: {coverage['completeness']:.0%} Remaining Gaps: {coverage['gaps']}")
        return {
            "topic": topic,
            "results": results,
            "coverage": coverage
        }

    MAX_TOOL_ITERATIONS = 5 
    def _call_subagent(self, agent_name, prompt):
        agent_info = self.SUBAGENTS[agent_name]
        messages = [{"role": "user", "content": prompt}]

        for _ in range(self.MAX_TOOL_ITERATIONS):
            request={
                "model": self.model,
                "max_tokens": 4096,
                "system": agent_info["system_prompt"],
                "messages": messages                
            }
            if agent_info["tools"]:
                request["tools"] = agent_info["tools"]
            response = self.client.messages.create(**request)

            if response.stop_reason == "end_turn":
                content = "\n".join([block.text for block in response.content if block.type == "text"])
                return self._parse_json(content)

            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})
                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    handler = TOOL_HANDLERS.get(block.name)
                    try:
                        if handler is None:
                            raise ValueError(f"unknown tool '{block.name}'")
                        output, is_error= handler(**block.input), False 
                    except Exception as e:
                        output, is_error = f"Error: {str(e)}", True

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output,
                        "is_error": is_error
                    })
                messages.append({"role": "user", "content": tool_results})
                continue
            if response.stop_reason == "max_tokens":
                raise RuntimeError(f"Subagent {agent_name} reached max tokens without completing the task.")
            raise RuntimeError(f"{agent_name} Unexpected stop reason: {response.stop_reason}")
        raise RuntimeError(f"Subagent {agent_name} exceeded max tool iterations ({self.MAX_TOOL_ITERATIONS}) without completing the task.")


    MIN_SEARCH_RESULTS = 3
    def evaluate_coverage(self, topic, subtopics,results):
        search=results.get("web_search_agent", {})
        analysis=results.get("document_analysis_agent", {})
        coverage={}
        to_judge={}

        # Stage 1: a deterministic code check for missing or thin results
        for st in subtopics:
            n_results= len(search.get(st, {}).get("results", []))
            n_points= len(analysis.get(st, {}).get("analysis", []))
            if n_results==0:
                coverage[st]= {"status": "missing", "reason": "No search results found"}
            elif n_results<self.MIN_SEARCH_RESULTS or n_points==0:
                coverage[st]= {"status": "superficial", "reason": f"Only {n_results} search results, {n_points} analysis points found"}            
            else:
                to_judge[st]= analysis[st]["analysis"]
        
        # Stage 2: the coordinator judges the rest against each subtopic's scope
        if to_judge:
            judge_prompt= (
                f"Overall research goal: {topic}\n\n"
                f"Findings for each subtopic: {json.dumps(to_judge)}\n\n"
                "For each subtopic, judge whether the findings cover it substantively. "
                "Judge only against the subtopic's own scope: if a subtopic names several things "
                "(for example 'X and Y'), each one must have substantive findings. "
                "Ignore gaps outside the subtopic's scope, such as cost or policy for a "
                "technology subtopic; these are covered elsewhere or out of scope.\n"
                "Use the subtopic names exactly as given. "
                'Reply with ONLY a JSON object, with no other text: '
                '{"<subtopic>": {"status": "substantive" or "superficial", "reason": "..."}}'
            )
            verdict=self._ask_coordinator(judge_prompt)
            for st in to_judge:
                coverage[st] = verdict.get(st, {"status": "superficial", "reason": "No verdict provided"})

        gaps= {st for st, info in coverage.items() if info["status"] != "substantive"}
        completeness= (len(subtopics) - len(gaps))/len(subtopics) if subtopics else 0
        return {"coverage": coverage, "gaps": gaps, "completeness": completeness}
        

if __name__ == "__main__":
    coordinator = HubSpokeResearchCoordinator()
    topic = "The impact of climate change on global agriculture"
    topic="renewable energy technologies"
    summary = coordinator.research(topic,required_areas=["solar", "wind", "geothermal", "tidal", "biomass", "fusion"])
    print(summary)
    
