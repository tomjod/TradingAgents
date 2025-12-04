import time
import json


def create_neutral_debator(llm):
    def neutral_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        neutral_history = risk_debate_state.get("neutral_history", "")

        current_risky_response = risk_debate_state.get("current_risky_response", "")
        current_safe_response = risk_debate_state.get("current_safe_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]

        prompt = f"""You are the NEUTRAL Risk Analyst. Your role is to find the optimal risk/reward balance.

YOUR STANCE: Seek the middle ground with data-driven balance. Challenge both extremes.

TRADER'S CURRENT DECISION:
{trader_decision}

YOUR TASK:
1. EVALUATE both the upside (risky) and downside (safe) perspectives
2. CHALLENGE both the aggressive AND conservative analysts where they're wrong
3. Propose a BALANCED approach that captures upside while limiting downside
4. End with a CONCRETE recommendation: "SCALE IN with [X]% now, [Y]% at [level]" or "HEDGE with [instrument]"

DATA TO USE:
- Market Report: {market_research_report}
- Sentiment: {sentiment_report}
- News: {news_report}
- Fundamentals: {fundamentals_report}

DEBATE CONTEXT:
{history}
Risky Analyst's Last Point: {current_risky_response}
Safe Analyst's Last Point: {current_safe_response}

RULES:
- Be DECISIVE, not wishy-washy
- Use DATA to justify your balanced view
- End with ONE clear, actionable compromise
- Speak conversationally, no special formatting"""

        response = llm.invoke(prompt)

        argument = f"Neutral Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "risky_history": risk_debate_state.get("risky_history", ""),
            "safe_history": risk_debate_state.get("safe_history", ""),
            "neutral_history": neutral_history + "\n" + argument,
            "latest_speaker": "Neutral",
            "current_risky_response": risk_debate_state.get(
                "current_risky_response", ""
            ),
            "current_safe_response": risk_debate_state.get("current_safe_response", ""),
            "current_neutral_response": argument,
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return neutral_node
