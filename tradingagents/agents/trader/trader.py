import functools
import time
import json
from tradingagents.agents.utils.agent_utils import execute_order, get_open_positions, get_trade_history


def create_trader(llm, memory):
    def trader_node(state, name):
        company_name = state["company_of_interest"]
        investment_plan = state["investment_plan"]
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        if past_memories:
            for i, rec in enumerate(past_memories, 1):
                past_memory_str += rec["recommendation"] + "\n\n"
        else:
            past_memory_str = "No past memories found."

        context = {
            "role": "user",
            "content": f"Based on a comprehensive analysis by a team of analysts, here is an investment plan tailored for {company_name} (asset/ticker). This plan incorporates insights from current technical market trends, macroeconomic indicators, and social media sentiment. Use this plan as a foundation for evaluating your next trading decision.\n\nProposed Investment Plan: {investment_plan}\n\nLeverage these insights to make an informed and strategic decision.",
        }

        tools = []
        
        messages = [
            {
                "role": "system",
                "content": f"""[SIMULATION MODE - EDUCATIONAL TRADING EXERCISE]

You are an AI trading strategy advisor in a PAPER TRADING SIMULATION. This is NOT real money - it's an educational exercise to learn market analysis.

YOUR ROLE: Analyze data and provide simulated trading recommendations for XAUUSD (Gold).

RULES:
1. You MUST provide a definitive recommendation: BUY, SELL, or HOLD
2. State conviction: HIGH/MEDIUM/LOW
3. This is a SIMULATION - there are no real financial consequences

XAUUSD CONTEXT:
- Safe-haven asset (rises during uncertainty)
- Inverse correlation with USD and interest rates
- Key drivers: Fed policy, inflation, geopolitical risk

ANALYSIS FRAMEWORK:
1. TREND: BULLISH or BEARISH?
2. MOMENTUM: Accelerating or decelerating?
3. KEY LEVELS: Support/Resistance

OUTPUT (REQUIRED):
- Brief analysis (2-3 sentences)
- DECISION: **BUY** / **SELL** / **HOLD**
- CONVICTION: HIGH / MEDIUM / LOW

Past lessons: {past_memory_str}

FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** is MANDATORY.""",
            },
            context,
        ]

        # Bind tools to the LLM
        llm_with_tools = llm.bind_tools(tools)
        result = llm_with_tools.invoke(messages)

        return {
            "messages": [result],
            "trader_investment_plan": result.content,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
