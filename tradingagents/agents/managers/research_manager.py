import time
import json


def create_research_manager(llm, memory):
    def research_manager_node(state) -> dict:
        history = state["investment_debate_state"].get("history", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        investment_debate_state = state["investment_debate_state"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""You are the Portfolio Manager for a HIGH-FREQUENCY SCALPING BOT making IMMEDIATE trading decisions.

⚠️ SCALPING CONTEXT:
- We hold positions for MINUTES (5-30 min max)
- Target: 50-300 points profit per trade
- We use M5 (5-minute) timeframe
- This is NOT buy and hold - we need quick, decisive signals

CRITICAL RULES:
1. You MUST choose BUY, SELL, or HOLD. No ambiguity.
2. BUY = "Enter LONG scalp NOW"
3. SELL = "Enter SHORT scalp NOW"
4. HOLD = "No clear scalping opportunity right now"
5. NEVER say "If you believe", "Consider", "You might". State facts and conclusions.
6. Your decision is FINAL - the scalping bot will act on it IMMEDIATELY.

SCALPING DECISION PROCESS:
1. Review the bull vs bear debate
2. Identify which side has STRONGER short-term momentum arguments
3. Make a DEFINITIVE choice for the NEXT 5-15 MINUTES
4. Assign conviction: HIGH (>75%), MEDIUM (50-75%), LOW (<50%)

OUTPUT REQUIREMENTS:
- Summarize key SCALPING signals from BOTH sides (2-3 sentences each)
- State your DECISION: **BUY** / **SELL** / **HOLD**
- State CONVICTION level
- Provide 1 clear RATIONALE sentence

Past scalping lessons (USE THESE TO AVOID REPEATING MISTAKES):
\"{past_memory_str}\"

DEBATE TO ANALYZE:
{history}"""
        response = llm.invoke(prompt)

        new_investment_debate_state = {
            "judge_decision": response.content,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": response.content,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": response.content,
        }

    return research_manager_node
