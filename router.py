"""The router: decides, for every message, whether to answer, ask, or call a tool.

There is no separate classifier. Gemini's function calling IS the router: the model
reads the tool descriptions and the system prompt and picks an action. This file
controls everything around that decision:
  - the system prompt (the behavioural contract),
  - the tool-calling loop (capped, logged, errors contained),
  - a grounding check that catches numbers the tools never produced.
"""
import json
import re
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation

from google.genai import types

import config
import products
import services
import assess_flotation

def _assess_flotation(silica_concentrate: float, iron_concentrate: float) -> dict:
    return assess_flotation.assess_flotation(
        silica_concentrate, iron_concentrate, csv_path=str(config.FLOTATION_CSV))

# def _assess_flotation(silica_concentrate: float, iron_concentrate: float,
#                       deeper_research: bool = False) -> dict:
#     return assess_flotation.assess_flotation(
#         silica_concentrate, iron_concentrate,
#         deeper_research=deeper_research, csv_path=str(config.FLOTATION_CSV))

# ---------------------------------------------------------------- tools
TOOL_FUNCS = {
    "search_brochures": products.search_brochures,
    "estimate_service_cost": services.estimate_service_cost,
    "assess_flotation": _assess_flotation,
}

DECLARATIONS = [
    types.FunctionDeclaration(
        name="search_brochures",
        description=(
            "Search the Metso product brochures. Use for ANY question about products, specifications, "
            "features, applications, or when recommending a product. This is the only valid source of "
            "product facts. Returns text passages with brochure name and page."
        ),
        parameters={
            "type": "OBJECT",
            "properties": {
                "query": {"type": "STRING",
                          "description": "A specific, self-contained search query in English, including the customer's situation."},
                "product": {"type": "STRING",
                            "description": "Optional. Restrict to one brochure by product name (use names from the available product list)."},
                "diverse": {"type": "BOOLEAN",
                            "description": "Set true when recommending or comparing: returns at most one passage per brochure."},
            },
            "required": ["query"],
        },
    ),
    types.FunctionDeclaration(
        name="estimate_service_cost",
        description=(
            "Estimate the cost of a field service visit from historical service reports. Requires equipment, "
            "service type and country. If one is missing, the tool says which; ask the user for it. "
            "Never estimate costs without this tool."
        ),
        parameters={
            "type": "OBJECT",
            "properties": {
                "equipment": {"type": "STRING", "description": "Equipment model, e.g. 'Larox PF-DS'."},
                "service_type": {"type": "STRING", "description": "The work, e.g. 'filter cloth replacement'."},
                "country": {"type": "STRING", "description": "Country where the visit takes place."},
                "quantity": {"type": "NUMBER", "description": "Optional. Number of items serviced, e.g. 12 cloths."},
            },
            "required": [],   # deliberately not required here: the tool validates and tells us what's missing
        },
    ),
        types.FunctionDeclaration(
        name="assess_flotation",
        description=(
            "Assess a customer’s flotation concentrate quality against benchmark plant data: where their"
            "silica and iron grades sit. Use when the customer gives concentrate grades or asks about"
            "flotation quality. Never assess concentrate quality without this tool."
            # "Assess a customer’s flotation concentrate quality against benchmark plant data: where their"
            # "silica and iron grades sit. Use when the customer gives concentrate grades or asks about flotation"
            # "quality. Never assess concentrate quality without this tool. Call it again with deeper_research=true"
            # "only if the salesperson asks for deeper research; that adds the dosing-timing analysis and the extra"
            # "data to request."
        ),
        parameters={
            "type": "OBJECT",
            "properties": {
                "silica_concentrate": {"type": "NUMBER", "description": "Customer's % silica in the concentrate, e.g. 3.1."},
                "iron_concentrate": {"type": "NUMBER", "description": "Customer's % iron in the concentrate, e.g. 64.5."},
                # "deeper_research": {"type": "BOOLEAN",
                #                     "description": "Set true ONLY after the salesperson has said they want deeper research. "
                #                                    "Then the result includes the extra data to request from the customer."},
            },
            "required": ["silica_concentrate", "iron_concentrate"],
        },
    ),
]


def build_system_prompt() -> str:
    names = "\n".join(f"- {n}" for n in products.product_names())
    return f"""You are a sales assistant for a Metso salesperson who is on a live call with a mining customer.
Be concise and practical: short paragraphs, no filler.

TOOLS AND SOURCES OF TRUTH
- Product facts (specifications, features, applications) come ONLY from search_brochures results.
  Never use your own general knowledge for product facts, numbers or claims.
- Cost figures come ONLY from estimate_service_cost. Never invent or adjust them.
- Quote numbers exactly as the tools return them. Do not round, convert or recompute them.
- Flotation findings (benchmarks) come ONLY from assess_flotation. Never diagnose the customer's numbers from general knowledge.

WHAT TO DO
1. Product questions: search the brochures (rephrase and search again if the first result is weak).
   Cite the brochure and page for every specification, like (Brochure name, p. 4).
2. Recommendations: name ONE specific product and give the reason tied to what the customer said.
   Mention important caveats or things the brochure does not confirm.
3. Cost questions: call estimate_service_cost. If the tool says all comparable cases are from other countries, say so in your first sentence. Present a range (min-max, median), how many past
   reports it is based on, the report IDs, the currency, and every caveat the tool returns.
   Call it a rough estimate based on past visits. If a currency has n = 1, say 'one past visit: X' instead of min, max and median.
4. Missing information: if the tool reports missing_fields, or the request is too vague to search
   meaningfully, ask ONE short question for exactly what you need. Do not guess.
5. Not in the data: if a tool returns no_confident_match, data_unavailable, unknown_product or no_comparable_cases, say
   plainly that you do not have that information, and say what you do have. Do not fill the gap.
6. Anything unrelated to products, service costs or flotation concentrate quality: say it is outside what you can help with.
7. Flotation / concentrate quality: call assess_flotation with the grades the customer gave; if one is
   missing, ask ONE short question. Answer ONLY from its benchmark: where the customer's grades sit
   against the benchmark plant (share of hours with lower silica and higher iron, expected silica at
   that iron grade, median silica in hours with similar iron). Give a plain verdict on whether the
   grades look normal for that benchmark. Mention the limitations briefly, including that the benchmark
   is a different plant. Do not discuss dosing, timing, data quality or causes, and never promise an
   improvement.

AVAILABLE BROCHURES
{names}
"""

# 8. If the salesperson says yes to deeper research, call assess_flotation again with the same grades and
#    deeper_research=true. Then explain the timing finding (say "consistent with reactive dosing" ONLY if
#    verdict is consistent_with_reactive_dosing; otherwise say the data does not show a clear direction),
#    give its limitations, and list at most 4 of the most relevant data_requests, each with its reason.


# ---------------------------------------------------------------- grounding check
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _norm_num(s: str) -> str | None:
    try:
        return format(Decimal(s.replace(",", "")).normalize(), "f")
    except InvalidOperation:
        return None


def numbers_in(text: str) -> set[str]:
    return {n for n in (_norm_num(m) for m in _NUM.findall(text)) if n}


def ungrounded_numbers(answer: str, tool_outputs: list[dict], user_text: str) -> list[str]:
    """Numbers in the answer that appear in neither a tool result nor the user's own words.

    Single-digit whole numbers are ignored (list numbering, 'one of', etc.).
    Limitation: it checks presence, not meaning. It catches invented figures, not misread ones.
    """
    allowed = numbers_in(json.dumps(tool_outputs, default=float)) | numbers_in(user_text)
    return sorted(n for n in numbers_in(answer) if n not in allowed and not (n.isdigit() and len(n) == 1))


# ---------------------------------------------------------------- the agent
class Agent:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.llm_calls = 0   # watch this against the daily request quota
        self.client = config.get_client()
        self.history: list[types.Content] = []
        # Automatic function calling is OFF so that WE run the loop: we can cap it, log it and check it.
        # (Temperature is left at the default on purpose; Gemini 3 models are tuned for it.)
        self.gen_config = types.GenerateContentConfig(
            system_instruction=build_system_prompt(),
            tools=[types.Tool(function_declarations=DECLARATIONS)],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        config.LOG_DIR.mkdir(exist_ok=True)
        self.trace_path = config.LOG_DIR / "trace.jsonl"

    # -- plumbing -------------------------------------------------------
    def _generate(self, force_text: bool = False):
        cfg = self.gen_config
        if force_text:   # last allowed step: forbid further tool calls so we always end on text
            cfg = cfg.model_copy(update={"tool_config": types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="NONE"))})
        self.llm_calls += 1
        if self.verbose:
            print(f"  [llm call #{self.llm_calls} this session]")
        return config.api_call("agent", "gen", lambda: self.client.models.generate_content(
            model=config.GEN_MODEL, contents=self.history, config=cfg))

    def _log(self, record: dict):
        record["ts"] = datetime.now(timezone.utc).isoformat()
        with self.trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=float) + "\n")

    def _call_tool(self, name: str, args: dict) -> dict:
        func = TOOL_FUNCS.get(name)
        try:
            result = func(**args) if func else {"status": "error", "message": f"unknown tool {name}"}
        except Exception as e:   # a failing tool must not crash the conversation
            result = {"status": "error", "message": f"{type(e).__name__}: {e}"}
        result = json.loads(json.dumps(result, default=float))   # make numpy types plain JSON
        if self.verbose:
            print(f"  [tool] {name}({json.dumps(args, ensure_ascii=False)}) -> {result.get('status')}")
        self._log({"tool": name, "args": args, "result": result})
        return result

    def _user_text(self) -> str:
        return " ".join(p.text for c in self.history if c.role == "user" for p in (c.parts or []) if p.text)

    # -- the loop -------------------------------------------------------
    def _run_until_text(self, tool_outputs: list[dict]) -> str:
        for step in range(config.MAX_TOOL_STEPS):
            resp = self._generate(force_text=(step == config.MAX_TOOL_STEPS - 1))
            if not resp.candidates or not resp.candidates[0].content:
                return "(The model returned no answer. Please rephrase.)"
            # Append the model's content untouched: Gemini 3 needs its thought signatures echoed back.
            self.history.append(resp.candidates[0].content)

            calls = resp.function_calls or []
            if not calls:
                return resp.text or ""

            parts = []
            for fc in calls:
                result = self._call_tool(fc.name, dict(fc.args or {}))
                tool_outputs.append(result)
                parts.append(types.Part(function_response=types.FunctionResponse(
                    id=fc.id, name=fc.name, response={"result": result})))
            self.history.append(types.Content(role="user", parts=parts))
        return "(Stopped after too many tool steps.)"

    def respond(self, user_text: str) -> str:
        self.history.append(types.Content(role="user", parts=[types.Part(text=user_text)]))
        tool_outputs: list[dict] = []
        answer = self._run_until_text(tool_outputs)

        bad = ungrounded_numbers(answer, tool_outputs, self._user_text())
        if bad:
            self._log({"grounding_retry": bad})
            self.history.append(types.Content(role="user", parts=[types.Part(text=(
                f"SYSTEM CHECK: these figures in your answer do not appear in any tool result: {', '.join(bad)}. "
                "Rewrite the answer using only figures from the tool results, or remove them."))]))
            answer = self._run_until_text(tool_outputs)
            bad = ungrounded_numbers(answer, tool_outputs, self._user_text())
            if bad:
                answer += f"\n\n[Unverified figures, please double-check: {', '.join(bad)}]"
        return answer