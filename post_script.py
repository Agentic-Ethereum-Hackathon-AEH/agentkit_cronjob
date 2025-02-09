#!/usr/bin/env python
"""
AgentKit Expanded Flow Example

This script demonstrates a robust on-chain agent that:
  1. Fetches posts from an external API with pagination.
  2. Performs advanced dynamic analysis to compute an AI rating (0–5)
     based on sentiment, originality, bias, readability, and plagiarism.
  3. Saves post ratings externally via a POST request to a ratings endpoint.
  4. Processes all posts in a paginated fashion, computes a weighted final rating,
     ranks the posts, and distributes tokens to the top 10 posts via a simulated batch multicall.
  5. Checks wallet balances and initiates token payments.

Requirements:
  - spacy (with "en_core_web_sm" model)
  - textstat
  - transformers
  - scikit-learn
  - ollama
  - requests
  - qdrant-client, sentence-transformers
  - python-dotenv
  - cdp_langchain, langgraph
"""

import os
import sys
import time
import json
import math
import re
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any

import spacy
import textstat
import requests
import ollama

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from dotenv import load_dotenv
# --- Import official message types from langchain.schema ---
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_ollama.chat_models import ChatOllama

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from cdp_langchain.agent_toolkits import CdpToolkit
from cdp_langchain.utils import CdpAgentkitWrapper
from cdp_langchain.tools import CdpTool
from pydantic import BaseModel, Field

# --- Additional imports for plagiarism check ---
import hashlib
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# ------------------------------------------------
# MONKEY PATCH: Force unknown message types into HumanMessage
# ------------------------------------------------
import langchain_ollama
_orig_create_chat_stream = langchain_ollama.chat_models.ChatOllama._create_chat_stream

def _patched_create_chat_stream(self, messages, stop, **kwargs):
    patched_messages = []
    for m in messages:
        if isinstance(m, (HumanMessage, AIMessage, SystemMessage)):
            patched_messages.append(m)
        else:
            # Convert any unsupported message into a HumanMessage
            content = getattr(m, "content", str(m))
            patched_messages.append(HumanMessage(content=content))
    return _orig_create_chat_stream(self, patched_messages, stop, **kwargs)

langchain_ollama.chat_models.ChatOllama._create_chat_stream = _patched_create_chat_stream

# ------------------------------------------------
# Global Initialization: NLP, Logging, Embedding, Qdrant, etc.
# ------------------------------------------------
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
except Exception as e:
    print("Error loading spaCy model 'en_core_web_sm':", e)
    sys.exit(1)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize embedding model (for plagiarism check)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
def get_embedding(text: str) -> list:
    return embedding_model.encode(text).tolist()

# Initialize Qdrant client and create collection if needed.
QDRANT_COLLECTION_NAME = "text_embeddings"
client = QdrantClient(":memory:")
try:
    client.get_collection(QDRANT_COLLECTION_NAME)
except Exception:
    client.create_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

def check_plagiarism(text: str) -> Dict[str, float]:
    """Check for plagiarism using vector similarity with Qdrant."""
    try:
        embedding = get_embedding(text)
        results = client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=embedding,
            limit=1
        )
        text_hash = hashlib.md5(text.encode()).hexdigest()
        client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=[
                PointStruct(
                    id=text_hash,
                    vector=embedding,
                    payload={"text": text}
                )
            ]
        )
        if results:
            return {"score": round(1 - results[0].score, 2)}
        return {"score": 0.0}
    except Exception as e:
        logger.error(f"Plagiarism check failed: {str(e)}")
        return {"score": 0.0}

# ------------------------------------------------
# Global Configuration & Constants
# ------------------------------------------------
load_dotenv()
wallet_data_file = "wallet_data.txt"
POST_API_URL = os.environ.get("POST_API_URL", "http://localhost:3000/api/posts")
MAX_PAGES = int(os.environ.get("MAX_PAGES", "100"))
TOKEN_CONTRACT_ADDRESS = os.environ.get(
    "TOKEN_CONTRACT_ADDRESS", 
    "0x51C5D3bC02924Ca790E76F8464eF0c7ae54aC704"
)
# New constant for saving ratings via HTTP
RATINGS_API_URL = os.environ.get("RATINGS_API_URL", "http://localhost:3000/api/ratings/ai")
# Total tokens available for distribution among top 10 posts
TOTAL_TOKENS = 10000

# ------------------------------------------------
# Tool B: Fetch Posts (with Pagination)
# ------------------------------------------------
class FetchArticlesInput(BaseModel):
    page: int = Field(1, description="Page number (each page returns 50 posts).")

def fetch_articles(wallet=None, page: int = 1) -> str:
    url = f"{POST_API_URL}?page={page}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        posts = response.json()
        return json.dumps(posts)
    except Exception as e:
        logger.error(f"Error fetching posts on page {page}: {e}")
        return json.dumps([])

# ------------------------------------------------
# Tool C: Simple Article Analysis (Heuristic)
# ------------------------------------------------
class AnalyzeArticleInput(BaseModel):
    article_content: str = Field(..., description="The content of the article to analyze.")

def analyze_article(article_content: str) -> str:
    """Return 4.5 if 'great' is in the text; otherwise, 3.0."""
    return str(4.5 if "great" in article_content.lower() else 3.0)

# ------------------------------------------------
# Advanced Analysis Functions (Dynamic AI Rating)
# ------------------------------------------------
# Global LLM instance will be set during agent initialization.
LLM_INSTANCE = None

def analyze_sentiment(text: str) -> dict:
    """Analyze text sentiment using a transformers pipeline."""
    from transformers import pipeline
    sentiment_analyzer = pipeline("sentiment-analysis")
    result = sentiment_analyzer(text)[0]
    return {"label": result["label"], "score": result["score"]}

def detect_bias(text: str) -> dict:
    """
    Detect bias using the global LLM instance via a plain prompt.
    Returns a dictionary with 'score' from 0 to 1.
    """
    global LLM_INSTANCE
    if LLM_INSTANCE is None:
        raise Exception("LLM_INSTANCE is not set.")
    prompt = f"Analyze bias in this text. Respond ONLY with a numerical score (0-1, 0=neutral, 1=biased):\n\n{text}"
    response = LLM_INSTANCE.generate([HumanMessage(content=prompt)])
    # Extract the generated text using double indexing
    text_output = response.generations[0][0].message.content
    match = re.search(r"0?\.?\d+", text_output)
    return {"score": float(match.group()) if match else 0.5}

def evaluate_originality(text: str) -> dict:
    """Evaluate originality using TF-IDF similarity."""
    corpus = ["AI is transforming industries.", "Bias in AI is a concern."]
    vectorizer = TfidfVectorizer().fit(corpus + [text])
    vectors = vectorizer.transform(corpus + [text])
    similarity = cosine_similarity(vectors[-1], vectors[:-1]).mean()
    return {"similarity_score": similarity, "score": 1 - similarity}

def classify_topics(text: str) -> dict:
    """
    Classify main and secondary topics using the global LLM via a plain prompt.
    """
    global LLM_INSTANCE
    if LLM_INSTANCE is None:
        raise Exception("LLM_INSTANCE is not set.")
    prompt = (
        "Analyze the main and secondary topics of this text.\n"
        "Return EXACTLY in this format:\n"
        "Main:topic|Secondary:topic1|Secondary:topic2\n\n"
        f"Text:\n{text}"
    )
    response = LLM_INSTANCE.generate([HumanMessage(content=prompt)])
    # Extract text using double indexing
    response_text = response.generations[0][0].message.content.strip()
    main_topic = "Unknown"
    secondary_topics = []
    if "Main:" in response_text:
        parts = response_text.split("|")
        for part in parts:
            if ":" in part:
                label, value = part.split(":", 1)
                if label.strip().lower() == "main":
                    main_topic = value.strip()
                elif label.strip().lower() == "secondary":
                    if value.strip() not in secondary_topics:
                        secondary_topics.append(value.strip())
    if not secondary_topics:
        doc = nlp(text)
        topics = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "GPE", "TECH"]]
        secondary_topics = list(set(topics))[:2]
    return {"main_topic": main_topic, "secondary_topics": secondary_topics[:2]}

def evaluate_readability(text: str) -> dict:
    """Calculate readability scores."""
    return {
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "gunning_fog_index": textstat.gunning_fog(text)
    }

def dynamic_analysis(text: str) -> dict:
    """
    Run several analysis functions concurrently and return a summary.
    Includes sentiment, bias, originality, plagiarism, readability, and topics.
    """
    doc = nlp(text)
    cleaned_text = " ".join([token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct])
    with ThreadPoolExecutor() as executor:
        future_sentiment = executor.submit(analyze_sentiment, cleaned_text)
        future_bias = executor.submit(detect_bias, cleaned_text)
        future_originality = executor.submit(evaluate_originality, cleaned_text)
        future_topics = executor.submit(classify_topics, cleaned_text)
        future_readability = executor.submit(evaluate_readability, cleaned_text)
        future_plagiarism = executor.submit(check_plagiarism, text)

        sentiment = future_sentiment.result()
        bias = future_bias.result()
        originality = future_originality.result()
        topics = future_topics.result()
        readability = future_readability.result()
        plagiarism = future_plagiarism.result()

    return {
        "sentiment": {"label": sentiment["label"], "score": round(float(sentiment["score"]), 2)},
        "bias": {"score": round(float(bias["score"]), 2)},
        "originality": {"score": round(float(originality["score"]), 2)},
        "plagiarism": {"score": round(float(plagiarism["score"]), 2)},
        "readability": {
            "flesch_kincaid_grade": round(float(readability["flesch_kincaid_grade"]), 1),
            "gunning_fog_index": round(float(readability["gunning_fog_index"]), 1)
        },
        "topic": {"main_topic": topics["main_topic"], "secondary_topics": topics["secondary_topics"]}
    }

def compute_final_rating(text: str) -> str:
    """
    Compute a final AI rating (0-5) by combining:
      - Sentiment (converted to a 0-5 scale),
      - Originality (0-5 scale),
      - A penalty for bias,
      - A penalty for difficult readability.
    """
    analysis = dynamic_analysis(text)
    sentiment = analysis.get("sentiment", {})
    originality = analysis.get("originality", {})
    bias = analysis.get("bias", {})
    readability = analysis.get("readability", {})

    sentiment_label = sentiment.get("label", "").lower()
    sentiment_score = sentiment.get("score", 0.5)
    if sentiment_label == "positive":
        sentiment_rating = sentiment_score * 5
    elif sentiment_label == "negative":
        sentiment_rating = 5 - (sentiment_score * 5)
    else:
        sentiment_rating = 2.5

    originality_rating = originality.get("score", 0.5) * 5
    bias_penalty = bias.get("score", 0.5) * 1.0
    fk_grade = readability.get("flesch_kincaid_grade", 10)
    readability_penalty = max(0, fk_grade - 10) * 0.1

    ai_rating = ((sentiment_rating + originality_rating) / 2) - (bias_penalty + readability_penalty)
    ai_rating = max(0, min(ai_rating, 5))
    return str(round(ai_rating, 2))

# ------------------------------------------------
# Tool D: Advanced Article Analysis (Dynamic AI Rating)
# ------------------------------------------------
class AnalyzeArticleAdvancedInput(BaseModel):
    article_content: str = Field(..., description="The content of the article to analyze.")

def analyze_article_advanced(article_content: str) -> str:
    """Compute a dynamic AI rating (0-5) from the article content."""
    return compute_final_rating(article_content)

# ------------------------------------------------
# Tool E: Save Article Rating (Simulated External API)
# ------------------------------------------------
class SaveArticleRatingInput(BaseModel):
    post_id: int = Field(..., description="The post ID.")
    user_rating: float = Field(..., description="User rating (0-5).")
    ai_rating: float = Field(..., description="AI computed rating (0-5).")
    final_rating: float = Field(..., description="Weighted final rating (0-5).")

def save_article_rating(data: dict) -> str:
    """
    POST the rating data to an external API endpoint.
    This function ensures the required field 'rating' is present by deriving it from final_rating.
    It also adds dummy values for other required fields and nests the post connection.
    """
    if "rating" not in data:
        data["rating"] = int(round(data["final_rating"]))
    dummy_fields = {
        "sentimentAnalysisLabel": "",
        "sentimentAnalysisScore": 0.0,
        "biasDetectionScore": 0.0,
        "biasDetectionDirection": "",
        "originalityScore": 0.0,
        "similarityScore": 0.0,
        "readabilityFleschKincaid": 0.0,
        "readabilityGunningFog": 0.0,
        "mainTopic": "",
        "secondaryTopics": []
    }
    for key, value in dummy_fields.items():
        if key not in data:
            data[key] = value
    if "post_id" in data and data["post_id"]:
        data["post"] = {"connect": {"id": data["post_id"]}}
    try:
        response = requests.post(RATINGS_API_URL, json=data, timeout=10)
        response.raise_for_status()
        return f"Rating saved: {response.text}"
    except Exception as e:
        return f"Error saving rating: {str(e)}"

# ------------------------------------------------
# Tool F: Check Wallet Balance
# ------------------------------------------------
class CheckWalletBalanceInput(BaseModel):
    asset_id: str = Field(..., description="Asset identifier (e.g. 'eth' or 'usdc').")

def check_wallet_balance(wallet, asset_id: str) -> str:
    """Check the wallet balance for the specified asset."""
    try:
        balances = {}
        for address in wallet.addresses:
            balance = address.balance(asset_id)
            balances[address.address_id] = str(balance)
        return json.dumps(balances, indent=2)
    except Exception as e:
        return f"Error checking balance: {str(e)}"

# ------------------------------------------------
# Tool G: Initiate Token Payment
# ------------------------------------------------
class InitiatePaymentInput(BaseModel):
    recipient: str = Field(..., description="Recipient's wallet address.")
    amount: str = Field(..., description="Amount to pay in wei.")

def initiate_token_payment(wallet, recipient: str, amount: str) -> str:
    """Initiate a token payment from the wallet to a recipient."""
    try:
        payment_tx = wallet.transfer(amount=amount, asset_id="eth", destination=recipient).wait()
        return (
            f"Payment of {amount} wei sent to {recipient}.\n"
            f"Tx hash: {payment_tx.transaction.transaction_hash}\n"
            f"Tx link: {payment_tx.transaction.transaction_link}"
        )
    except Exception as e:
        return f"Error initiating payment: {str(e)}"

# ------------------------------------------------
# Helper: Batch Token Distribution (Simulated Multicall)
# ------------------------------------------------
def batch_distribute_tokens(wallet, distribution_calls: list) -> str:
    """Simulate a batch distribution call."""
    if not distribution_calls:
        return "No distributions to process."
    results = []
    for dist in distribution_calls:
        user_address = dist.get("user_address")
        tokens = dist.get("tokens", 0)
        results.append(f"Distributed {tokens} tokens to {user_address}")
    return "Batch distribution completed:\n" + "\n".join(results)

# ------------------------------------------------
# Tool H: Process Articles Flow (Full Cron Workflow with Pagination)
# ------------------------------------------------
class ProcessArticlesFlowInput(BaseModel):
    page: int = Field(1, description="Starting page number (each page contains 50 posts).")

def process_articles_flow(wallet, dummy_input: str = "", page: int = 1) -> str:
    """
    Full workflow:
      1. Loop through pages to fetch posts.
      2. For each post, compute a final rating:
         final_rating = (userRating * 0.6 + AI Rating * 0.4)
         where AI Rating is computed dynamically.
      3. Save the rating externally (via POST to the ratings endpoint).
      4. Collect distribution calls.
      5. Rank posts by final rating and select the top 10.
      6. Distribute a fixed TOTAL_TOKENS proportionally among the top 10.
    Returns a summary report.
    """
    summary_lines = []
    distribution_calls = []
    current_page = page

    while current_page <= MAX_PAGES:
        summary_lines.append(f"Fetching posts from page {current_page}...")
        articles_json = fetch_articles(page=current_page)
        try:
            articles = json.loads(articles_json)
        except Exception as e:
            summary_lines.append(f"Error parsing posts on page {current_page}: {str(e)}")
            break

        if not articles:
            summary_lines.append(f"No posts found on page {current_page}. Ending pagination.")
            break

        for article in articles:
            post_id = article.get("id", 0)
            title = article.get("title", "No Title")
            content = article.get("content", "")
            user_address = article.get("user_address", "")
            user_rating = float(article.get("userRating", 3.0))
            ai_rating = float(analyze_article_advanced(content))
            final_rating = (user_rating * 0.6) + (ai_rating * 0.4)
            final_rating = round(final_rating, 2)

            summary_lines.append(
                f"Post {post_id}: '{title}'\nUser Rating: {user_rating} | AI Rating: {ai_rating} | Final Rating: {final_rating}"
            )

            save_data = {
                "post_id": post_id,
                "user_rating": user_rating,
                "ai_rating": ai_rating,
                "final_rating": final_rating
            }
            save_result = save_article_rating(save_data)
            summary_lines.append(f"Saved Rating: {save_result}")

            if final_rating > 0:
                # Store the final_rating so we can proportionally allocate tokens later
                distribution_calls.append({
                    "user_address": user_address,
                    "final_rating": final_rating
                })

        if len(articles) < 50:
            summary_lines.append(f"Fetched only {len(articles)} posts on page {current_page}. Ending pagination.")
            break
        current_page += 1

    # Select top 10 posts by final rating
    top_distributions = sorted(distribution_calls, key=lambda x: x["final_rating"], reverse=True)[:10]
    # Calculate total rating among top posts
    total_rating = sum(item["final_rating"] for item in top_distributions)
    # Allocate tokens proportionally (using TOTAL_TOKENS constant)
    for item in top_distributions:
        item["tokens"] = int((item["final_rating"] / total_rating) * TOTAL_TOKENS) if total_rating > 0 else 0

    summary_lines.append("\nInitiating token distribution for the top 10 posts...")
    batch_result = batch_distribute_tokens(wallet, top_distributions)
    summary_lines.append(batch_result)

    return "\n".join(summary_lines)

# ------------------------------------------------
# Additional Tool: Distribute and Mint Token (Individual)
# ------------------------------------------------
class DistributeTokenInput(BaseModel):
    user_address: str = Field(..., description="The address to receive tokens.")
    tokens: int = Field(..., description="The number of tokens to mint and distribute.")

def distribute_and_mint_token(wallet, user_address: str, tokens: int) -> str:
    """Simulate minting and distributing tokens to a single user."""
    return f"Minted and distributed {tokens} tokens to {user_address}."

# ------------------------------------------------
# Agent Initialization: Register All Tools as CDP Actions
# ------------------------------------------------
def initialize_agent():
    global LLM_INSTANCE
    llm = ChatOllama(model="llama3.2:latest")
    LLM_INSTANCE = llm

    wallet_data = None
    if os.path.exists(wallet_data_file):
        with open(wallet_data_file) as f:
            wallet_data = f.read()
    values = {"cdp_wallet_data": wallet_data} if wallet_data else {}
    agentkit = CdpAgentkitWrapper(**values)
    updated_wallet_data = agentkit.export_wallet()
    with open(wallet_data_file, "w") as f:
        f.write(updated_wallet_data)

    cdp_toolkit = CdpToolkit.from_cdp_agentkit_wrapper(agentkit)
    base_tools = cdp_toolkit.get_tools()

    fetchArticlesTool = CdpTool(
        name="fetch_articles",
        description="Fetch a list of posts from the external API with pagination support.",
        cdp_agentkit_wrapper=agentkit,
        args_schema=FetchArticlesInput,
        func=fetch_articles,
    )
    analyzeArticleTool = CdpTool(
        name="analyze_article_advanced",
        description="Perform advanced dynamic analysis on an article to compute an AI rating (0-5).",
        cdp_agentkit_wrapper=agentkit,
        args_schema=AnalyzeArticleAdvancedInput,
        func=analyze_article_advanced,
    )
    distributeTokenTool = CdpTool(
        name="distribute_and_mint_token",
        description="Mint tokens to a user based on rating (tokens = rating * 100).",
        cdp_agentkit_wrapper=agentkit,
        args_schema=DistributeTokenInput,
        func=distribute_and_mint_token,
    )
    processArticlesFlowTool = CdpTool(
        name="process_articles_flow",
        description=(
            "Execute the full workflow: fetch posts (with pagination), analyze each post using advanced dynamic analysis, "
            "save ratings externally, rank posts by weighted rating (userRating*0.6 + AI Rating*0.4), "
            "and distribute tokens (tokens = final rating * 100) to the top 10 posts via batch multicall."
        ),
        cdp_agentkit_wrapper=agentkit,
        args_schema=ProcessArticlesFlowInput,
        func=process_articles_flow,
    )
    checkBalanceTool = CdpTool(
        name="check_wallet_balance",
        description="Check the wallet balance for a specified asset (e.g. 'eth' or 'usdc').",
        cdp_agentkit_wrapper=agentkit,
        args_schema=CheckWalletBalanceInput,
        func=check_wallet_balance,
    )
    initiatePaymentTool = CdpTool(
        name="initiate_token_payment",
        description="Initiate a token payment from the wallet to a recipient.",
        cdp_agentkit_wrapper=agentkit,
        args_schema=InitiatePaymentInput,
        func=initiate_token_payment,
    )
    saveRatingTool = CdpTool(
        name="save_article_rating",
        description="Save the computed article rating to an external API (simulated).",
        cdp_agentkit_wrapper=agentkit,
        args_schema=SaveArticleRatingInput,
        func=save_article_rating,
    )

    all_tools = base_tools + [
        fetchArticlesTool,
        analyzeArticleTool,
        distributeTokenTool,
        processArticlesFlowTool,
        checkBalanceTool,
        initiatePaymentTool,
        saveRatingTool,
    ]

    memory = MemorySaver()

    state_modifier = (
        "You are a robust on-chain agent using Coinbase Developer Platform AgentKit. "
        "Your capabilities include:\n"
        " - Fetching posts with pagination (fetch_articles).\n"
        " - Performing advanced dynamic analysis on posts to compute an AI rating (analyze_article_advanced).\n"
        " - Saving computed ratings externally (save_article_rating).\n"
        " - Ranking posts and distributing tokens in a batch multicall (process_articles_flow) based on the formula: tokens = final rating * 100, "
        "where final rating = (userRating * 0.6 + AI Rating * 0.4).\n"
        " - Checking wallet balances (check_wallet_balance) and initiating token payments (initiate_token_payment).\n\n"
        "When instructed to 'process articles' or similar, call the process_articles_flow tool and return its result."
    )

    config = {"configurable": {"thread_id": "CDP Agentkit Expanded Flow Example"}}

    agent_executor = create_react_agent(
        llm,
        tools=all_tools,
        checkpointer=memory,
        state_modifier=state_modifier,
    )

    return agentkit, agent_executor, config

# ------------------------------------------------
# Automatic Process Execution (Non-interactive)
# ------------------------------------------------
def run_automatic_process(agentkit, config):
    print("Starting automatic process execution...\n")
    result = process_articles_flow(agentkit.wallet)
    print("=== Full Process Flow Summary ===")
    print(result)
    print("=== End of Process ===")

# ------------------------------------------------
# Main Function: Automatic Process Execution
# ------------------------------------------------
def main():
    agentkit, agent_executor, config = initialize_agent()
    run_automatic_process(agentkit, config)

if __name__ == "__main__":
    print("Starting AgentKit Expanded Flow with Automatic Process Execution...")
    main()
