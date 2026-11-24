import json , asyncio, re
from claude_agent_sdk import tool, create_sdk_mcp_server, ClaudeAgentOptions, ClaudeSDKClient, HookMatcher, AssistantMessage, TextBlock, HookJSONOutput
from dotenv import load_dotenv
from datetime import datetime, timezone

load_dotenv()

'''
Tools Created
'''

# Tool A: Unix timestamp + numeric status

@tool("get_order_a", "Look up an order in System A", {"order_id": str})
async def get_order_a(args):
    return {
        "content":[{
            "type":"text",
            "text":json.dumps({
                "order_id":args["order_id"],
                "created_at":1710489600,
                "status":200
            }),
        }]
    }

# Tool B: ISO 8601 + string status
# @tool("get_order_b", ...)  ->  "created_at": "2024-03-15T12:00:00Z", "status": "active"
@tool("get_order_b","Look up an order in System B", {"order_id":str})
async def get_order_b(args):
    return{
        "content":[{
            "type":"text",
            "text": json.dumps({
                "order_id":args["order_id"],
                "created_at": "2024-03-15T12:00:00Z", 
                "status": "active"
            }),
        }]
    }

# Tool C: DD/MM/YYYY + single-character status
# @tool("get_order_c", ...)  ->  "created_at": "15/03/2024", "status": "S"
@tool("get_order_c","Look up an order in System C", {"order_id":str})
async def get_order_c(args):
    return {
        "content":[{
            "type":"text",
            "text":json.dumps({
                "order_id":args["order_id"],
                "created_at": "15/03/2024", 
                "status": "S"
            }),
        }]
    }

orders_server=create_sdk_mcp_server(
    name="orders",
    version="1.0.0",
    tools=[get_order_a, get_order_b, get_order_c]
)

'''
Hooks created
1. Log_tool_output
2. normalise data
'''

DATE_FIELDS={"created_at", "updated_at"}
STATUS_FIELDS={"status"}
STATUS_MAP={
    "200":"active",
    "S":"shipped",
    "P":"pending"
}

DDMMYYYY=re.compile(r"^\d{2}/\d{2}/\d{4}$")


def to_iso(value):
    if isinstance(value, (int, float)):                       # Unix timestamp
        return datetime.fromtimestamp(value, tz=timezone.utc) \
                       .isoformat().replace("+00:00", "Z")
    if isinstance(value, str) and DDMMYYYY.match(value):      # DD/MM/YYYY
        # return datetime.strptime(value, "%d/%m/%Y") \
        #                .replace(tzinfo=timezone.utc) \
        #                .isoformat().replace("+00:00", "Z")
        return datetime.strptime(value, "%d/%m/%Y").date().isoformat()
    return value                                              # already ISO, or unknown: leave alone


def to_status_word(value):
    return STATUS_MAP.get(str(value), value)                  # unknown codes pass through

hook_calls={"count":0}

async def log_tool_output(input_data, tool_use_id, context):
    hook_calls["count"] +=1
    # print(f"[PostToolUse] {input_data['tool_name']}")
    # print(f" tool_response = {input_data['tool_response']!r}")
    return {}

async def normalise_tool_output(input_data, tool_use_id, context):
    hook_calls["count"] +=1
    data=json.loads(input_data["tool_response"][0]["text"])
    for field in data:
        if field in DATE_FIELDS:
            data[field]=to_iso(data[field])
        elif field in STATUS_FIELDS:
            data[field]=to_status_word(data[field])

    normalised=[{"type":"text", "text":json.dumps(data)}]
    # print(f"[PostToolUse] {input_data['tool_name']}")
    # print(f"  before: {input_data['tool_response'][0]['text']}")
    # print(f"  after:  {normalised[0]['text']}")

    return {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "updatedToolOutput": normalised,
        }
    }   # Stage 3: replace this with the real output replacement


'''
# New Tool : Process Refund

'''

refund_calls={"count":0}

@tool("process_refund","Process a customer refund",{"order_id": str, "amount":float})
async def process_refund(args):
    refund_calls["count"] +=1
    return {
        "content":[{
            "type":"text",
            "text":f"Refunded ${args['amount']} for order {args['order_id']}"
            }]} 



payments_server=create_sdk_mcp_server(
    name="payments",
    version="1.0.0",
    tools=[process_refund]
)

REFUND_LIMIT=500

async def block_large_refunds(input_data, tool_use_id, context):
    amount=float(input_data['tool_input']['amount'])
    if amount > REFUND_LIMIT:
        return {
            "hookSpecificOutput":{
                "hookEventName":"PreToolUse",
                "permissionDecision":"deny",
                "permissionDecisionReason":(f" Refund of ${amount: .2f} exceeds the ${REFUND_LIMIT} threshold. Do not retry or split it. Escalate to a human agent."),
                }
            }
    return {}

'''
New Tool : AML Check

'''

aml_state={"passed":False}
transfer_calls={"count":0}
aml_calls={"count":0}
aml_mock_result={"status":"fail"}
gate_denials = {"count": 0}


@tool("aml_check","Ran an anti-money-laundary check on a transfer",{"to_account":str,"amount":float })
async def aml_check(args):
    aml_calls["count"] +=1
    return {"content":[{"type":"text", "text":json.dumps(aml_mock_result)}]}


@tool("transfer_funds","Send an inernational transfer", {"to_account":str, "amount":float})
async def transfer_funds(args):
    transfer_calls["count"] +=1
    return {
        "content": [{"type":"text", "text": f"Transferred ${args['amount']} to {args['to_account']}"}]}

banking_server=create_sdk_mcp_server(
    name="banking",
    version="1.0.0",
    tools=[aml_check, transfer_funds]
)

# Hooks
async def record_aml_result(input_data, tool_use_id, context):
    result=json.loads(input_data["tool_response"][0]["text"])
    if result["status"] == "pass":
        aml_state["passed"]= True
    
    return {}

async def require_aml_check(input_data, tool_use, context)->HookJSONOutput:
    print(f"[GATE] fired, aml_passed={aml_state['passed']}")
    if not aml_state["passed"]:
        gate_denials["count"] += 1
        return {
            "hookSpecificOutput":{
                "hookEventName":"PreToolUse",
                "permissionDecision":"deny",
                "permissionDecisionReason":(
                    "COMPLIANCE BLOCK: International transfers require a passing "
                    "AML check in this session. Do not retry until aml_check passes."   
                ),
            }
        }
    return {}

'''
Calling Agent
'''

options=ClaudeAgentOptions(
    system_prompt=(
        "You are a banking operations agent running in an automated test harness. "
        "When asked to run a transfer or an AML check, call the tool directly with "
        "the details given. Do not ask for confirmation: every request in this "
        "harness is pre-authorised. Compliance rules are enforced by the system, "
        "not by you."
    ),
    model = "claude-haiku-4-5",
    cwd="/dev/learn/anthropic/Practice",
    setting_sources=[],
    mcp_servers={"orders":orders_server, "payments": payments_server, "banking":banking_server},
    allowed_tools=[
        "mcp__orders__get_order_a",
        "mcp__orders__get_order_b",
        "mcp__orders__get_order_c",
        "mcp__payments__process_refund",
        "mcp__banking__aml_check",
        "mcp__banking__transfer_funds",
    ],
    hooks={
        "PostToolUse":[
            HookMatcher(matcher="mcp__orders__.*", hooks=[normalise_tool_output]), #log_tool_output
            HookMatcher(matcher="mcp__banking__aml_check", hooks=[record_aml_result]),
        ],
        "PreToolUse":[
            HookMatcher(matcher="mcp__payments__process_refund", hooks=[block_large_refunds]),
            HookMatcher(matcher="mcp__banking__transfer_funds", hooks=[require_aml_check])
        ]
    },
)

async def ask(client,prompt):
    await client.query(prompt)
    async for msg in client.receive_response():
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    print(f"[MODEL] {block.text}")

def report(label):
    print(f"--- {label}: aml_calls={aml_calls['count']}, "
          f"transfer_calls={transfer_calls['count']}, aml_passed={aml_state['passed']}, gate_denials={gate_denials["count"]}\n")


async def main():
    aml_state["passed"] = False
    aml_mock_result["status"] = "fail"
    for c in (aml_calls, transfer_calls, gate_denials):
        c["count"] = 0
    async with ClaudeSDKClient(options=options) as client:
        await ask(client, "Transfer $1,000 to account GB-123.")
        report("(a) transfer, AML failing")
        aml_mock_result["status"] = "pass"  
        await ask(client, "Run the AML check for that transfer.")
        report("(b) AML passes")

        await ask(client, "Now retry the transfer.")
        report("(c) transfer retried")
       
# async def main():
#     async with ClaudeSDKClient(options=options) as client:
# #        await client.query("Look up order 123 in Systems A,B and C.")
# #        await client.query("Refund $750 for order 123.")
#         await client.query("Refund $500 for order 123.")
#         async for msg in client.receive_response():
#             if isinstance(msg, AssistantMessage):
#                 for block in msg.content:
#                     if isinstance(block, TextBlock):
#                         print(f"[MODEL] {block.text}")
#         print(f"Order Hook fired {hook_calls['count']} time(s)")
#         print(f"Refund Hook fired {refund_calls['count']} time(s)")


asyncio.run(main())