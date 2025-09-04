from fastapi import FastAPI, UploadFile, File
from agentic_doc import parse

app = FastAPI()

@app.get("/")
def root():
    return {"message": "agentic-doc is running"}

@app.post("/parse")
async def parse_document(file: UploadFile = File(...)):
    content = await file.read()
    result = parse.parse(content)
    return {"result": result}
