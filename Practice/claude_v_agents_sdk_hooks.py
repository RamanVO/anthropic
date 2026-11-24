"""
Reference implementation: Agent SDK hooks for normalisation and policy enforcement.
Claude Certified Architect (Foundations), Domain 1, Task Statement 1.5.

  PostToolUse  -> transforms results AFTER execution, BEFORE the model sees them
  PreToolUse   -> blocks calls BEFORE execution

Run:  python hooks_reference.py
"""

import asyncio
import json
import re
from datetime import datetime, timezone

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookJSONOutput,
    HookMatcher,
    TextBlock,
    create_sdk_mcp_server,
    tool,
)

# ---------------------------------------------------------------------------
# Shared state (module-level; reset() puts it back to a known state per test)
# ---------------------------------------------------------------------------
counts = {"normalise": 0, "refund": 0, "refund_denied": 0,
          "aml": 0, "transfer": 0, "transfer_denied": 0}
aml_cleared: set[tuple[str, float]] = set()   # (account, amount) pairs that passed AML
aml_mock_result = {"status": "fail"}          # test lever for the aml_check tool
violations: list[str] = []                    # transfers that ran without clearance


def reset() -> None:
    for k in counts:
        counts[k] = 0
    aml_cleared.clear()
    aml_mock_result["status"] = "fail"
    violations.clear()


def text_result(payload) -> dict:
    return {"content": [{"type": "text", "text": json.dumps(payload)}]}


# ---------------------------------------------------------------------------
# 1. Three order tools with heterogeneous formats (server: orders)
# ---------------------------------------------------------------------------
@tool("get_order_a", "Look up an order in System A", {"order_id": str})
async def get_order_a(args):
    return text_result({"order_id": args["order_id"], "created_at": 1710489600, "status": 200})


@tool("get_order_b", "Look up an order in System B", {"order_id": str})
async def get_order_b(args):
    return text_result({"order_id": args["order_id"],
                        "created_at": "2024-03-15T12:00:00Z", "status": "active"})


@tool("get_order_c", "Look up an order in System C", {"order_id": str})
async def get_order_c(args):
    return text_result({"order_id": args["order_id"], "created_at": "15/03/2024", "status": "S"})


orders_server = create_sdk_mcp_server(
    name="orders", version="1.0.0", tools=[get_order_a, get_order_b, get_order_c])


# ---------------------------------------------------------------------------
# 2. PostToolUse normaliser (allowlist of fields; never invents precision)
# ---------------------------------------------------------------------------
DATE_FIELDS = {"created_at", "updated_at"}
STATUS_FIELDS = {"status"}
STATUS_MAP = {"200": "active", "S": "shipped", "P": "pending"}
DDMMYYYY = re.compile(r"^\d{2}/\d{2}/\d{4}$")


def to_iso(value):
    if isinstance(value, (int, float)):                      # Unix timestamp: real time
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, str) and DDMMYYYY.match(value):     # date only: keep it date only
        return datetime.strptime(value, "%d/%m/%Y").date().isoformat()
    return value                                             # already ISO, or unknown


def to_status_word(value):
    return STATUS_MAP.get(str(value), value)


async def normalise_tool_output(input_data, tool_use_id, context) -> HookJSONOutput:
    counts["normalise"] += 1
    data = json.loads(input_data["tool_response"][0]["text"])          # parse
    for field in data:                                                  # normalise
        if field in DATE_FIELDS:
            data[field] = to_iso(data[field])
        elif field in STATUS_FIELDS:
            data[field] = to_status_word(data[field])
    normalised = [{"type": "text", "text": json.dumps(data)}]          # repackage, same shape
    return {"hookSpecificOutput": {"hookEventName": "PostToolUse",
                                   "updatedToolOutput": normalised}}


# ---------------------------------------------------------------------------
# 4. Refund tool + PreToolUse threshold gate (server: payments)
# ---------------------------------------------------------------------------
REFUND_LIMIT = 500   # policy: refunds that EXCEED $500 go to a human, so ">" not ">="


@tool("process_refund", "Process a customer refund", {"order_id": str, "amount": float})
async def process_refund(args):
    counts["refund"] += 1                     # proof: counts real executions only
    return text_result({"refunded": args["amount"], "order_id": args["order_id"]})


payments_server = create_sdk_mcp_server(name="payments", version="1.0.0", tools=[process_refund])


async def block_large_refunds(input_data, tool_use_id, context) -> HookJSONOutput:
    amount = float(input_data["tool_input"]["amount"])
    if amount > REFUND_LIMIT:
        counts["refund_denied"] += 1
        return {"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": (
                f"Refund of ${amount:.2f} exceeds the ${REFUND_LIMIT} threshold. "
                "Do not retry or split it. Escalate to a human agent."),
        }}
    return {}


# ---------------------------------------------------------------------------
# 5. AML prerequisite (server: banking)
#    PostToolUse on aml_check RECORDS; PreToolUse on transfer_funds ENFORCES.
#    Clearance is per (account, amount), not one session-wide flag, so one
#    pass cannot unlock unrelated transfers.
# ---------------------------------------------------------------------------
def txn_key(tool_input) -> tuple[str, float]:
    return (str(tool_input["to_account"]), float(tool_input["amount"]))


@tool("aml_check", "Run an anti-money-laundering check on a transfer",
      {"to_account": str, "amount": float})
async def aml_check(args):
    counts["aml"] += 1
    return text_result(aml_mock_result)


@tool("transfer_funds", "Send an international transfer", {"to_account": str, "amount": float})
async def transfer_funds(args):
    counts["transfer"] += 1
    if txn_key(args) not in aml_cleared:      # invariant check, independent of the hook
        violations.append(f"transfer ran without AML clearance: {args}")
    return text_result({"transferred": args["amount"], "to_account": args["to_account"]})


banking_server = create_sdk_mcp_server(
    name="banking", version="1.0.0", tools=[aml_check, transfer_funds])


async def record_aml_result(input_data, tool_use_id, context) -> HookJSONOutput:
    result = json.loads(input_data["tool_response"][0]["text"])
    if result.get("status") == "pass":
        aml_cleared.add(txn_key(input_data["tool_input"]))
    return {}


async def require_aml_check(input_data, tool_use_id, context) -> HookJSONOutput:
    if txn_key(input_data["tool_input"]) not in aml_cleared:
        counts["transfer_denied"] += 1
        return {"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": (
                "COMPLIANCE BLOCK: this transfer requires a passing AML check for the "
                "same account and amount. Do not retry until aml_check passes."),
        }}
    return {}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
options = ClaudeAgentOptions(
    model="claude-haiku-5-5",
    setting_sources=[],          # isolate from any project CLAUDE.md / settings
    system_prompt=(
        "You are an operations agent running in an automated test harness. When asked to "
        "look up orders, refund, run an AML check or transfer funds, call the tool directly "
        "with the details given. Do not ask for confirmation: every request here is "
        "pre-authorised. Compliance rules are enforced by the system, not by you."),
    mcp_servers={"orders": orders_server, "payments": payments_server, "banking": banking_server},
    allowed_tools=[
        "mcp__orders__get_order_a", "mcp__orders__get_order_b", "mcp__orders__get_order_c",
        "mcp__payments__process_refund",
        "mcp__banking__aml_check", "mcp__banking__transfer_funds",
    ],
    hooks={
        "PostToolUse": [
            HookMatcher(matcher="mcp__orders__.*", hooks=[normalise_tool_output]),
            HookMatcher(matcher="mcp__banking__aml_check", hooks=[record_aml_result]),
        ],
        # For enforcement hooks, consider on_failure="block" on the matcher
        # (Python Agent SDK v0.2.177+) so a crashing hook blocks, not allows.
        "PreToolUse": [
            HookMatcher(matcher="mcp__payments__process_refund", hooks=[block_large_refunds]),
            HookMatcher(matcher="mcp__banking__transfer_funds", hooks=[require_aml_check]),
        ],
    },
)


# ---------------------------------------------------------------------------
# Tests: assert on tool execution counts and invariants, not hook return values
# ---------------------------------------------------------------------------
async def ask(client, prompt):
    await client.query(prompt)
    async for msg in client.receive_response():
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    print(f"  [MODEL] {block.text}")


def check(label, condition):
    print(f"{'PASS' if condition else 'FAIL'}  {label}   {counts}")
    assert condition, label


async def test_normalisation():
    reset()
    async with ClaudeSDKClient(options=options) as client:
        await ask(client, "Look up order 123 in systems A, B and C.")
    check("normaliser fired for all three tools", counts["normalise"] == 3)
    assert to_iso(1710489600) == "2024-03-15T08:00:00Z"
    assert to_iso("15/03/2024") == "2024-03-15"            # no invented midnight
    assert to_iso("2024-03-15T12:00:00Z") == "2024-03-15T12:00:00Z"
    assert [to_status_word(v) for v in (200, "S", "P", "active")] == \
           ["active", "shipped", "pending", "active"]


async def test_refunds():
    for amount, expect_executed in ((2500, False), (500, True), (200, True)):
        reset()
        async with ClaudeSDKClient(options=options) as client:
            await ask(client, f"Refund ${amount} for order 123.")
        if expect_executed:
            check(f"${amount} refund executes", counts["refund"] == 1)
        else:
            check(f"${amount} refund attempted and denied",
                  counts["refund_denied"] >= 1 and counts["refund"] == 0)


async def test_aml():
    reset()
    async with ClaudeSDKClient(options=options) as client:
        await ask(client, "Transfer $1,000 to account GB-123.")
        check("(a) transfer attempted, denied, nothing sent",
              counts["transfer_denied"] >= 1 and counts["transfer"] == 0)

        aml_mock_result["status"] = "pass"
        await ask(client, "Run the AML check for $1,000 to GB-123, then make that transfer.")
        check("(b/c) transfer executes after AML passes", counts["transfer"] >= 1)

        await ask(client, "Now transfer $5,000 to account FR-999.")
        check("(d) a different transaction is not unlocked by the earlier pass",
              counts["transfer_denied"] >= 2)

    # The rule itself, however proactive the model was:
    check("invariant: no transfer ever ran without clearance", not violations)


async def main():
    await test_normalisation()
    await test_refunds()
    await test_aml()
    print("\nAll checks passed.")


if __name__ == "__main__":
    asyncio.run(main())