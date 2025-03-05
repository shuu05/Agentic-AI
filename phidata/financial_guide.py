from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import os
from dotenv import load_dotenv
load_dotenv()

web_search_agent = Agent(
    name="web_search_agent",
    role="Search the web for the information",
    model=Groq(id="llama-3.3-70b-versatile "),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls = True,
    markdown=True,
)


finance_agent = Agent(
    name="financial AI agent",
    model=Groq(id="llama-3.3-70b-versatile "),
    tools=[YFinanceTools(stock_price=True,analyst_recommendations=True,stock_fundamentals=True,company_news=True)],
    instructions=["Use tables to display the data"],
    show_tool_calls = True,
    markdown=True,
)


multi_agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile "),
    team = [web_search_agent, finance_agent],
    instructions = ["Always include sources","Use tables to display the data"],
    show_tool_calls = True,
    markdown = True,
)

multi_agent.print_response("summarize analyst recommendation and share the latest news for NVDA",stream=True)