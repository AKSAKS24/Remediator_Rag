import os
import json
import datetime
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document

import re


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required.")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
CHROMA_DIR   = os.getenv("CHROMA_DIR", "chroma_rules")
RULESET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ruleset.txt")

app = FastAPI(
    title="ABAP Remediator (LLM + Chroma RAG, PwC tagging)",
    version="2.0"
)

class Unit(BaseModel):
    pgm_name: str
    inc_name: str
    type: str
    name: Optional[str] = ""
    class_implementation: Optional[str] = ""
    code: str
    llm_prompt: List[str] = Field(default_factory=list)

SYSTEM_MSG = (
    "You are a precise ABAP remediation engine. "
    "You receive two main sources of guidance: "
    "1. The authoritative ABAP standards ('retrieved_rules'), extracted from the company ruleset.txt file. "
    "2. The user-supplied 'llm_prompt' field, which contains a prioritized list of mandatory TODOs—this is your master instruction set or 'god bible'. "
    "You must ALWAYS read, interpret, and APPLY all TODOs listed in 'llm_prompt' to the user's code, making the necessary modifications directly in the ABAP code. "
    "NEVER simply restate or return the TODOs themselves—your goal is to return the corrected code. "
    "If there is any conflict, the instructions in 'llm_prompt' take precedence over the ruleset standards. "
    "Use the extracted rules for style, best practices, and ABAP correctness as a reference unless explicitly overridden by the TODOs. "
    "Return only STRICT JSON with the required output structure."
)

USER_TEMPLATE = """
<retrieved_rules>
{rules}
</retrieved_rules>

Remediate the following ABAP code using BOTH:
- The authoritative remediation rules and patterns above (from ruleset.txt).
- The 'llm_prompt' field BELOW (god bible).

Instructions:
- Implement ALL TODOs from 'llm_prompt' in the code remediation.
- If there is any conflict, obey the 'llm_prompt' over the extracted rules.
- Output the FULL remediated code (not a diff).
- Each ADDED or MODIFIED line must include:  " Added By Pwc{today_date}
- Use ECC-safe syntax unless TODOs allow otherwise.
- Return ONLY strict JSON with keys:
{{
  "remediated_code": "<full updated ABAP code with PwC comments>"
}}

Context:
- Retrieved rules: extracted from ruleset.txt via vector search
- Program: {pgm_name}
- Include: {inc_name}
- Unit type: {unit_type}
- Unit name: {unit_name}
- Today's date (PwC tag): {today_date}

Original ABAP code:
{code}

llm_prompt (bullets, to be fully implemented):
{llm_prompt_json}
""".strip()

prompt = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_MSG), ("user", USER_TEMPLATE)]
)

def today_iso() -> str:
    return datetime.date.today().isoformat()

def _extract_json_str(s: str) -> str:
    t = s.strip()
    if t.startswith("```"):
        t = t.split("```", 2)
        if len(t) == 3:
            t = t[1] if not t[1].lstrip().startswith("{") else t[1]
            t = "\n".join(line for line in t.splitlines() if not line.strip().lower().startswith("json")).strip()
        else:
            t = s
    return t.strip()

# --- WHOLE-RULE SPLITTER ---
def split_rules_by_heading(text: str) -> list:
    """
    Split ruleset.txt into a list of rule strings from each 'Rule N — ...' section.
    """
    # Matches each Rule section from "Rule X — ..." up to the next rule (or EOF)
    # Accepts both en-dash and em-dash, handles double line headings.
    pattern = r'(^Rule\s*\d+\s*[–—-].*?)(?=^Rule\s*\d+\s*[–—-]|\Z)'
    matches = re.findall(pattern, text, flags=re.MULTILINE | re.DOTALL)
    return [m.strip() for m in matches if m.strip()]

# --- RAG VECTOR STORE WITH WHOLE-RULE CHUNKING ---
class RulesetStore:
    """
    Loads/splits/embeds ruleset.txt by WHOLE RULE ("Rule N — ..."), hot reloads if changed.
    """
    def __init__(self, ruleset_path: str, chroma_dir: str, top_k=4):
        self.ruleset_path = ruleset_path
        self.chroma_dir = chroma_dir
        self.top_k = int(os.getenv("RAG_TOP_K", top_k))
        self.last_mtime = 0
        self.vectordb: Optional[Chroma] = None

    def load_or_reload(self):
        fpath = Path(self.ruleset_path)
        mtime = fpath.stat().st_mtime
        if self.vectordb is not None and self.last_mtime == mtime:
            return self.vectordb
        print(f"[RulesetStore] Loading/splitting/embedding WHOLE-RULE chunks (mtime={mtime})")
        with open(self.ruleset_path, encoding="utf-8") as f:
            full_text = f.read()
        rule_chunks = split_rules_by_heading(full_text)
        # Remove duplicates, empty
        seen = set()
        docs = []
        for c in rule_chunks:
            c = c.strip()
            if not c or c in seen:
                continue
            docs.append(Document(page_content=c))
            seen.add(c)
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        if Path(self.chroma_dir).exists():
            for file in Path(self.chroma_dir).glob("*"):
                try:
                    file.unlink()
                except Exception as e:
                    print("WARN: Failed to unlink chroma file:", e)
        vectordb = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=self.chroma_dir)
        self.last_mtime = mtime
        self.vectordb = vectordb
        return vectordb

    def get_relevant_rules(self, user_query: str, top_k: Optional[int] = None) -> List[str]:
        vectordb = self.load_or_reload()
        k = top_k if top_k is not None else self.top_k
        rel_docs = vectordb.similarity_search(query=user_query, k=k)
        return [doc.page_content for doc in rel_docs]

# Initialize RAG vector DB store (on startup)
ruleset_store = RulesetStore(RULESET_PATH, CHROMA_DIR)
ruleset_store = RulesetStore(RULESET_PATH, CHROMA_DIR, top_k=1)

@app.on_event("startup")
async def startup_event():
    print("Initializing RAG vector DB ...")
    await asyncio.to_thread(ruleset_store.load_or_reload)
    print("RAG vector DB ready.")

async def remediate_with_rag(unit: Unit) -> str:
    t0 = time.perf_counter()
    if not unit.llm_prompt:
        raise HTTPException(status_code=400, detail="llm_prompt must be a non-empty list.")
    search_query = "\n".join(unit.llm_prompt) or unit.code[:500]
    rule_chunks = await asyncio.to_thread(ruleset_store.get_relevant_rules, search_query)

 # >>>>>>> Print the relevant rules retrieved <<<<<<<
    print(f"\n[RAG] {len(rule_chunks)} rule(s) sent to LLM prompt:")
    for i, chunk in enumerate(rule_chunks):
        lines = [l for l in chunk.splitlines() if l.strip()]
        heading = lines[0] if lines else '<no heading>'
        print(f"\n--- Relevant Rule {i+1}: {heading} ---\n{chunk}\n{'-'*60}")
        
    if not rule_chunks:
        rule_chunks = ["No rules retrieved."]
    rules_text = "\n\n".join(rule_chunks)
    t1 = time.perf_counter()
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=1)
    payload = {
        "rules": rules_text,
        "pgm_name": unit.pgm_name,
        "inc_name": unit.inc_name,
        "unit_type": unit.type,
        "unit_name": unit.name or "",
        "code": unit.code or "",
        "today_date": today_iso(),
        "llm_prompt_json": json.dumps(unit.llm_prompt, ensure_ascii=False, indent=2),
    }
    msgs = prompt.format_messages(**payload)
    resp = await llm.ainvoke(msgs)
    t2 = time.perf_counter()
    content = resp.content or ""
    content = _extract_json_str(content)
    try:
        data = json.loads(content)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Model did not return valid JSON: {e}")

    rem = data.get("remediated_code", "")
    if not isinstance(rem, str) or not rem.strip():
        raise HTTPException(status_code=502, detail="Model returned empty or invalid 'remediated_code'.")

    t3 = time.perf_counter()
    print(f"[remediate_with_rag] Vector search: {(t1-t0):.3f}s | LLM: {(t2-t1):.3f}s | Parse: {(t3-t2):.3f}s | TOTAL: {(t3-t0):.3f}s")
    return rem

@app.post("/remediate")
async def remediate(unit: Unit) -> Dict[str, Any]:
    """
    Input JSON:
      {
        "pgm_name": "...",
        "inc_name": "...",
        "type": "...",
        "name": "",
        "class_implementation": "",
        "code": "<ABAP code>",
        "llm_prompt": [ "...bullet...", "...bullet..." ]
      }

    Output JSON:
      original fields + "rem_code": "<full remediated ABAP>"
    """
    rem_code = await remediate_with_rag(unit)
    obj = unit.model_dump()
    obj["rem_code"] = rem_code
    return obj    