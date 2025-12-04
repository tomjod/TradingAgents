import time
import json


def create_risky_debator(llm):
    def risky_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        risky_history = risk_debate_state.get("risky_history", "")

        current_safe_response = risk_debate_state.get("current_safe_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]

        prompt = f"""You are the AGGRESSIVE Risk Analyst. Your role is to advocate for HIGH-REWARD opportunities with conviction.

YOUR STANCE: Champion bold, high-upside trades. Challenge conservative views with data.

TRADER'S CURRENT DECISION:
{trader_decision}

YOUR TASK:
1. SUPPORT the bullish/aggressive elements of the trader's plan
2. COUNTER the conservative (Safe) and neutral analysts' arguments directly
3. Provide SPECIFIC reasons why higher risk = higher reward in this case
4. End with a CONCRETE recommendation: "INCREASE position size" or "ADD to position at [level]"

DATA TO USE:
- Market Report: {market_research_report}
- Sentiment: {sentiment_report}
- News: {news_report}
- Fundamentals: {fundamentals_report}

DEBATE CONTEXT:
{history}
Safe Analyst's Last Point: {current_safe_response}
Neutral Analyst's Last Point: {current_neutral_response}

RULES:
- Be ASSERTIVE, not tentative
- Use DATA to back every claim
- End with ONE clear, actionable statement
- Speak conversationally, no special formatting"""

        response = llm.invoke(prompt)

        argument = f"Risky Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "risky_history": risky_history + "\n" + argument,
            "safe_history": risk_debate_state.get("safe_history", ""),
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Risky",
            "current_risky_response": argument,
            "current_safe_response": risk_debate_state.get("current_safe_response", ""),
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return risky_node
