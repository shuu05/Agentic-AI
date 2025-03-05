from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import os
from dotenv import load_dotenv
import phi
from phi.playground import Playground,serve_playground_app

load_dotenv()
phi.api = os.getenv('PHI_API_Key')


web_search_agent = Agent(
    name="web_search_agent",
    role="Search the web for the information",
    model=Groq(id="llama-3.1-8b-instant"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls = True,
    markdown=True,
)


finance_agent = Agent(
    name="financial AI agent",
    model=Groq(id="llama-3.1-8b-instant"),
    tools=[YFinanceTools(stock_price=True,analyst_recommendations=True,stock_fundamentals=True,company_news=True)],
    instructions=["Use tables to display the data"],
    show_tool_calls = True,
    markdown=True,
)

app = Playground(agents=[web_search_agent, finance_agent]).get_app()

if __name__ == '__main__':
    serve_playground_app("playground:app",reload=True)