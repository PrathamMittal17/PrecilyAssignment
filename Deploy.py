# Importing the required libraries
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
pd.options.mode.chained_assignment = None

# Creating FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Importing the sentence transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Making the homepage endpoint
@app.get("/")
def root():
    return {"message": "Hello World"}


class Item(BaseModel):
    text1: str
    text2: str

# Making the endpoint for finding similarity between sentences
@app.post('/predict')
def predict(data: Item):
    embedding_1 = model.encode(data.text1)
    embedding_2 = model.encode(data.text2)
    similarity_score = cosine_similarity(embedding_1.reshape(1, -1), embedding_2.reshape(1, -1))[0][0]
    if similarity_score < 0:
        return {"similarity_score": 0}
    else:
        return {"similarity_score": np.float64(similarity_score)}
