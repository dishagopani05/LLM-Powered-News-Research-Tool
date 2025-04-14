from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

llm = GoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7
)
