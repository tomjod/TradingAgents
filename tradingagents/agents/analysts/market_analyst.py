from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import get_stock_data, get_indicators
from tradingagents.dataflows.config import get_config


def create_market_analyst(llm):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        tools = [
            get_stock_data,
            get_indicators,
        ]

        system_message = (
            """You are a DECISIVE Market Analyst for XAUUSD (Gold) trading. Your analysis MUST lead to actionable conclusions.

CRITICAL: Do NOT say "signals are mixed" or "it depends". State a PRIMARY TREND with conviction.

INDICATOR SELECTION (choose up to 8):

Moving Averages:
- close_50_sma: 50 SMA - medium-term trend
- close_200_sma: 200 SMA - long-term trend, golden/death cross
- close_10_ema: 10 EMA - short-term momentum

MACD:
- macd: Main MACD line
- macds: MACD Signal line
- macdh: MACD Histogram - momentum strength

Momentum:
- rsi: RSI - overbought/oversold (70/30)

Volatility:
- boll, boll_ub, boll_lb: Bollinger Bands - volatility and mean reversion
- atr: ATR - stop-loss sizing

Volume:
- vwma: Volume-weighted MA

ANALYSIS REQUIREMENTS:
1. State PRIMARY TREND: BULLISH / BEARISH / RANGING
2. State TREND STRENGTH: STRONG / MODERATE / WEAK
3. Identify KEY LEVELS: Support and Resistance
4. Give ACTIONABLE CONCLUSION: "Favor LONGS above X" or "Favor SHORTS below Y"

IMPORTANT: Call get_stock_data first, then get_indicators. Do NOT nest indicators.

XAUUSD CONTEXT: Gold correlates inversely with USD and yields. During risk-off, gold rises. Use this knowledge in your analysis.

End with a Markdown table summarizing: Indicator | Value | Interpretation | Bias (Bullish/Bearish/Neutral)"""
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. The asset/symbol we want to look at is {ticker}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(ticker=ticker)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content
       
        return {
            "messages": [result],
            "market_report": report,
        }

    return market_analyst_node
