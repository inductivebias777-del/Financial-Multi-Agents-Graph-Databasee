# agents.py (Corrected for Financial Domain)
"""
Defines the ADK agent team for the financial data application.
This includes a root agent for orchestration and specialized sub-agents
for graph querying, document retrieval, and stock price predictions.
"""
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from ..neo4j_for_adk import graphdb
from app.models.predict import predict_next_day_price
from langchain_google_vertexai import VertexAIEmbeddings

# --- Setup ---
llm = LiteLlm(model="gemini-2.5-flash")

embeddings = VertexAIEmbeddings(model_name="text-embedding-005")

# --- Tool Definitions ---
def query_graph_database(question: str) -> dict:
    """
    Generates a Cypher query for the financial graph and executes it.
    """
    schema = graphdb.send_query("CALL db.schema.visualization()")["query_result"]
    
    # CORRECTED: Updated schema description and examples to match actual database structure
    cypher_generation_prompt = f"""
    Task: Generate a Cypher statement to query a financial graph database.
    
    Schema: {schema}
    
    Instructions:
    - Use ONLY the provided relationship types and property keys.
    - The graph contains the following nodes and relationships:
      - (c:Company)-[:HAS_FINANCIALS]->(f:Financials)
      - (c:Company)-[:FILED]->(doc:Document)
      - (c:Company)-[:HAS_RISK]->(r:Risk)
      - (c:Company)-[:HAD_EVENT]->(e:Event)
      - (c:Company)-[:HAS_STRATEGY]->(s:Strategy)
      - (doc:Document)-[:MENTIONS_RISK]->(r:Risk)
      - (doc:Document)-[:DESCRIBES_EVENT]->(e:Event)
      - (doc:Document)-[:MENTIONS_STRATEGY]->(s:Strategy)
      - (chunk:Chunk) nodes with vector embeddings for document chunks
    
    - Key properties for nodes:
      - Company: `ticker` (e.g., 'NVDA'), `name`, `cik`
      - Financials: `company` (ticker), `year` (string like '2024'), `revenue`, `netIncome`, `eps`
      - Risk, Event, Strategy: `name`
      - Document: `source` (filename), `year`, `type`, `management_outlook`
      - Chunk: `text`, `embedding` (vector)
    
    - IMPORTANT: The Financials node uses `company` property (not ticker directly) and `year` is a STRING
    - Company tickers in your data: NVDA, MSFT, AAPL, GOOGL, AMZN
    
    Example Questions & Queries (ticker and year are database property names, not variables):
    - Question: "What was the revenue for NVDA in 2024?"
      Query: MATCH (c:Company {{ticker: 'NVDA'}})-[:HAS_FINANCIALS]->(f:Financials {{year: '2024'}}) RETURN f.revenue
    - Question: "What are the key risks for NVDA?"
      Query: MATCH (c:Company {{ticker: 'NVDA'}})-[:HAS_RISK]->(r:Risk) RETURN r.name
    - Question: "Show me financial trends for NVDA over the years"
      Query: MATCH (c:Company {{ticker: 'NVDA'}})-[:HAS_FINANCIALS]->(f:Financials) RETURN f.year, f.revenue, f.netIncome, f.eps ORDER BY f.year
    - Question: "What events happened at Apple?"
      Query: MATCH (c:Company {{ticker: 'AAPL'}})-[:HAD_EVENT]->(e:Event) RETURN e.name
    
    Question: {question}
    Return only the Cypher query, no explanation or formatting.
    """
    
    cypher_query = llm.llm_client.completion(
        model=llm.model,
        messages=[{"role": "user", "content": cypher_generation_prompt}],
        tools=[], # <-- ADD THIS LINE
    ).choices[0].message.content.strip()
    
    # Clean the response
    cypher_query = cypher_query.replace("```cypher", "").replace("```", "").strip()
    print(f"Generated Cypher: {cypher_query}")
    
    return graphdb.send_query(cypher_query)

def retrieve_from_documents(question: str) -> dict:
    """
    Performs vector search on 10-K filing chunks and synthesizes an answer.
    """
    question_embedding = embeddings.embed_query(question)
    
    search_query = """
    CALL db.index.vector.queryNodes('filings', 5, $embedding) YIELD node, score
    RETURN node.text AS text, score
    ORDER BY score DESC
    """
    
    search_results = graphdb.send_query(search_query, {"embedding": question_embedding})
    
    if search_results['status'] == 'error' or not search_results['query_result']:
        return {"answer": "Could not retrieve relevant documents from filings.", "error": search_results.get('message', 'Unknown error')}
    
    context = "\n".join([r['text'] for r in search_results['query_result']])
    
    synthesis_prompt = f"""
    Based on the following context from SEC 10-K filings, answer the question comprehensively.
    
    Context from filings:
    {context}
    
    Question: {question}
    
    Instructions:
    - Provide a detailed answer based on the context
    - If the context doesn't contain relevant information, say so
    - Cite specific information from the filings when possible
    - Focus on the financial and strategic aspects mentioned
    
    Answer:
    """
    
    response = llm.llm_client.completion(
        model=llm.model,
        messages=[{"role": "user", "content": synthesis_prompt}],
        tools=[], # <-- ADD THIS LINE
    ).choices[0].message.content
    
    return {"answer": response}

def predict_stock_price_tool(ticker: str) -> dict:
    """
    A wrapper for the stock price prediction model.
    Input must be a single, valid stock ticker string from our available companies.
    """
    valid_tickers = {'NVDA', 'MSFT', 'AAPL', 'GOOGL', 'AMZN'}
    
    if not isinstance(ticker, str):
        return {"error": f"Invalid input type. Please provide a ticker as a string."}
    
    ticker = ticker.upper().strip()
    
    if ticker not in valid_tickers:
        return {"error": f"Ticker '{ticker}' not found. Available tickers: {', '.join(valid_tickers)}"}
    
    print(f"Predicting price for ticker: {ticker}")
    return predict_next_day_price(ticker)

# --- Sub-Agent Definitions ---
graph_qa_subagent = Agent(
    name="GraphQA_Agent",
    model=llm,
    tools=[query_graph_database],
    description="Use for questions about company financials (revenue, net income, EPS), risks, events, strategies, and any structured data queries. Works with tickers: NVDA, MSFT, AAPL, GOOGL, AMZN.",
    instruction="""
    Your task is to use the `query_graph_database` tool to answer questions about:
    - Financial metrics (revenue, net income, EPS) by company and year
    - Company risks, events, and strategic focuses
    - Comparisons between companies
    - Financial trends over time
    
    Always use the exact ticker symbols: NVDA, MSFT, AAPL, GOOGL, AMZN
    Remember that years are stored as strings (e.g., '2024', '2023').
    """
)

document_rag_subagent = Agent(
    name="DocumentRAG_Agent",
    model=llm,
    tools=[retrieve_from_documents],
    description="Use for qualitative questions about company strategy, management outlook, detailed business descriptions, or any information that requires reading through SEC 10-K filing text.",
    instruction="""
    Your task is to use the `retrieve_from_documents` tool to find detailed, qualitative information from SEC filings including:
    - Management's discussion and analysis
    - Business strategy and outlook
    - Detailed risk descriptions
    - Product and service descriptions
    - Market analysis and competitive positioning
    
    Provide comprehensive answers based on the retrieved document chunks.
    """
)

prediction_subagent = Agent(
    name="StockPricePredictor_Agent",
    model=llm,
    tools=[predict_stock_price_tool],
    description="Use ONLY to predict the next day's closing stock price. Works with tickers: NVDA, MSFT, AAPL, GOOGL, AMZN.",
    instruction="""
    Your only task is to use the `predict_stock_price_tool` for stock price predictions.
    
    IMPORTANT:
    - Only valid tickers: NVDA, MSFT, AAPL, GOOGL, AMZN
    - Input must be a single ticker string
    - Always include a disclaimer that predictions are estimates based on historical data and not financial advice
    """
)

# --- Root Agent Definition ---
root_agent = Agent(
    name="Financial_Root_Agent",
    model=llm,
    sub_agents=[graph_qa_subagent, document_rag_subagent, prediction_subagent],
    description="The main financial assistant that analyzes user queries and delegates to specialized agents for financial data analysis.",
    instruction="""
    You are a knowledgeable financial data assistant with access to data for these companies: NVDA, MSFT, AAPL, GOOGL, AMZN.
    
    DELEGATION GUIDELINES:
    - Use 'GraphQA_Agent' for:
      * Specific financial numbers (revenue, net income, EPS)
      * Company risks, events, strategies (structured data)
      * Financial comparisons and trends
      * Any query requiring precise data extraction
    
    - Use 'DocumentRAG_Agent' for:
      * Qualitative analysis and detailed explanations
      * Management outlook and business strategy discussions
      * Complex business descriptions
      * Questions requiring reading through filing narratives
    
    - Use 'StockPricePredictor_Agent' ONLY for:
      * Explicit requests to predict future stock prices
      * Must use valid tickers: NVDA, MSFT, AAPL, GOOGL, AMZN
    
    IMPORTANT NOTES:
    - Available companies: NVIDIA (NVDA), Microsoft (MSFT), Apple (AAPL), Alphabet/Google (GOOGL), Amazon (AMZN)
    - Financial data years: 2021-2024
    - Always include disclaimers for predictions
    - If uncertain about which agent to use, explain your reasoning
    """
)