from langchain_core.messages import AIMessage
import time
import json


def create_safe_debator(llm):
    def safe_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        safe_history = risk_debate_state.get("safe_history", "")

        current_risky_response = risk_debate_state.get("current_risky_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]

        prompt = f"""You are the CONSERVATIVE Risk Analyst. Your role is to protect capital and minimize downside risk.

YOUR STANCE: Prioritize capital preservation. Challenge risky proposals with data.

TRADER'S CURRENT DECISION:
{trader_decision}

YOUR TASK:
1. IDENTIFY potential risks and downsides in the trader's plan
2. COUNTER the aggressive (Risky) and neutral analysts' arguments directly
3. Provide SPECIFIC risk factors: volatility, macro events, technical resistance
4. End with a CONCRETE recommendation: "REDUCE position size" or "SET tighter stop at [level]" or "WAIT for pullback to [level]"

DATA TO USE:
- Market Report: {market_research_report}
- Sentiment: {sentiment_report}
- News: {news_report}
- Fundamentals: {fundamentals_report}

DEBATE CONTEXT:
{history}
Risky Analyst's Last Point: {current_risky_response}
Neutral Analyst's Last Point: {current_neutral_response}

RULES:
- Be ASSERTIVE about risks, not fearful
- Use DATA to back every concern
- End with ONE clear, protective recommendation
- Speak conversationally, no special formatting"""

        response = llm.invoke(prompt)

        argument = f"Safe Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "risky_history": risk_debate_state.get("risky_history", ""),
            "safe_history": safe_history + "\n" + argument,
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Safe",
            "current_risky_response": risk_debate_state.get(
                "current_risky_response", ""
            ),
            "current_safe_response": argument,
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return safe_node
