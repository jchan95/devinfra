from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    SUPABASE_URL: str
    SUPABASE_KEY: str
    OPENAI_API_KEY: str
    GEMINI_API_KEY: Optional[str] = None
    
    class Config:
        env_file = ".env"

settings = Settings()