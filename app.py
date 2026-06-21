# Run Script: python -m uvicorn app:app --reload

from fastapi import FastAPI, Request          #!  main app framework & gives access to incoming HTTP request data (headers, body, etc.)
from pydantic import BaseModel                #! Used to define request/response data schemas and automatically validate data
from transformers import T5Tokenizer, Trainer, TrainingArguments, T5ForConditionalGeneration
import torch
import re
from fastapi.templating import Jinja2Templates #! Enables server-side HTML rendering using Jinja2 templates (NOT the UI itself, just rendering)
from fastapi.responses import HTMLResponse     #! Used to explicitly return HTML content as a response
from fastapi.staticfiles import StaticFiles    #! Used to serve static files (CSS, JS, images) from a directory


# initialized our fastapi app
app = FastAPI(title="Text Summmarizer App", description="Text Summarization using T5",version='1.0')

# model & tokenizer are loaded lazily in the startup event below,
# NOT at import time -- this lets uvicorn bind the port first.
model = None
tokenizer = None

# Device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    model.to(device)
    model.eval()  # inference mode -- avoids keeping autograd graph buffers around

# templating
templates = Jinja2Templates(directory=".")

#Input schema for dialogue => string format
class DialogueInput(BaseModel):
    dialogue:str

def clean_data(text):
    text= re.sub(r"\r\n"," ",text)   # next_line -> " "
    text= re.sub(r"\s+\n"," ",text)  # space
    text= re.sub(r"<.*?>"," ",text)  # html tag
    text=text.strip().lower()
    return text





def summarize_dialogue(dialogue:str)->str:
    dialogue = clean_data(dialogue) # clean_data
    
    # tokenizer
    inputs = tokenizer(
        dialogue,
        return_tensors="pt",  
        padding="max_length",
        max_length=512,
        truncation=True
    ).to(device)
    
    # generate the summary => token ids
    with torch.no_grad():  # no gradient tracking needed for inference -- saves memory
        target = model.generate(
            input_ids = inputs["input_ids"],
            attention_mask = inputs["attention_mask"],
            max_length = 150,
            num_beams = 4,  # generate 4 o/p and best one will be answer
            early_stopping = True
        )

    #token_id convert to summary --> decording
    summary = tokenizer.decode(target[0],skip_special_tokens=True)
    return summary




# API Endpoint
@app.post("/summarizer/")
async def summarizer(dialogue_input:DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return  {"summary":summary}

@app.get("/",response_class=HTMLResponse)
async def home(request:Request):
    return templates.TemplateResponse("index.html",{"request":request})
