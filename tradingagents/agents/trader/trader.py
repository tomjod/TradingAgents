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
                "content": f"""[HIGH-FREQUENCY SCALPING BOT - PAPER TRADING SIMULATION]

You are an AI trading strategy advisor for a HIGH-FREQUENCY SCALPING BOT. This is NOT buy and hold - we capture quick moves for small profits.

⚠️ SCALPING CONTEXT:
- We hold positions for MINUTES (5-30 min max)
- Target: 50-300 points profit per trade
- We use M5 (5-minute) timeframe
- Quick entries, quick exits
- Multiple small wins > fewer big wins

YOUR ROLE: Analyze data and provide IMMEDIATE scalping recommendations for XAUUSD (Gold).

RULES:
1. You MUST provide a definitive recommendation: BUY, SELL, or HOLD
2. State conviction: HIGH/MEDIUM/LOW
3. BUY = "Go LONG now for quick scalp"
4. SELL = "Go SHORT now for quick scalp"
5. HOLD = "No good scalping opportunity right now"

XAUUSD SCALPING TIPS:
- Best moves happen at session opens (London, NY)
- Quick pullbacks to EMA10 are scalp entries
- RSI extremes (>70 or <30) mean reverting opportunities
- Bollinger band touches = mean reversion scalps

ANALYSIS FRAMEWORK:
1. IMMEDIATE MOMENTUM: UP or DOWN?
2. ENTRY ZONE: Good price for scalp entry?
3. EXIT TARGET: Where to take profit (50-300 pts away)?

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
