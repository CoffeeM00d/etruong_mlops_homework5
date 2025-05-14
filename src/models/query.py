from pydantic import BaseModel
from typing import List, Dict

#class RAGRequest(BaseModel):
#    question: str
class RAGRequest(BaseModel):
    question: str
    num_responses: int = 2
    
class RAGResponse(BaseModel):
    answers: List[str]
