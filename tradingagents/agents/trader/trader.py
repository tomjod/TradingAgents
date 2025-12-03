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
                "content": f"""You are a Strategy Advisor for an automated trading system. Your role is to analyze market data and provide a directional bias (BUY, SELL, or HOLD) for the execution bot ("Soldier").

                CRITICAL: You do NOT execute trades yourself. You do NOT manage positions. Your ONLY job is to determine the best market direction based on the analysis provided.

                - Analyze the provided reports (Market, Sentiment, News, Fundamentals).
                - Determine the overall trend and conviction.
                - Recommend a clear directional bias:
                    - **BUY**: If the market is bullish and conditions favor long positions.
                    - **SELL**: If the market is bearish and conditions favor short positions.
                    - **HOLD**: If the market is uncertain or choppy.

                End with a firm decision and always conclude your response with 'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**' to confirm your recommendation. Do not forget to utilize lessons from past decisions to learn from your mistakes. Here is some reflections from similar situatiosn you traded in and the lessons learned: {past_memory_str}""",
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
