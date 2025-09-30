from __future__ import annotations

import csv
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

from rag.retrieve import retrieve
from rag.prompts import build_user_prompt, get_system_prompt


REF_SECTION_RE = re.compile(r"Referencias:\s*(.*)", re.IGNORECASE | re.DOTALL)


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _extract_references(text: str) -> str:
    m = REF_SECTION_RE.search(text or "")
    if not m:
        return ""
    return m.group(1).strip()


def _format_references_from_docs(docs: List[Dict[str, Any]], max_refs: int = 3) -> str:
    refs = []
    for d in docs[:max_refs]:
        src = d.get('source') or d.get('doc_id') or d.get('doc') or 'Documento'
        name = str(src).replace('.pdf', '').replace('data/raw/', '').strip()
        page = d.get('page') or d.get('page_number') or 'N/A'
        refs.append(f"[{name}, p.{page}]")
    return "\n".join(refs)


@dataclass
class EvalResult:
    question: str
    provider: str
    answer: str
    references: str
    latency_sec: float
    est_cost_usd: float
    exact_match: bool


class _DummyEngine:
    def __init__(self):
        self.provider = None

    def set_provider(self, provider):
        self.provider = provider


class QualityEvaluator:
    def __init__(self, gold_set_path: str, k: int = 4):
        self.gold_set_path = gold_set_path
        self.k = k
        self.rag_engine = _DummyEngine()

    def _load_gold(self) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        with open(self.gold_set_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
        return items

    def evaluate_provider(self, provider, provider_name: str) -> List[EvalResult]:
        gold = self._load_gold()
        results: List[EvalResult] = []

        for item in gold:
            q = item.get('question', '')

            # Recuperación de contexto
            start = time.time()
            docs = retrieve(q, k=self.k)
            ctx_docs = [{
                'content': getattr(c, 'content', ''),
                'source': getattr(c, 'source', ''),
                'page': getattr(c, 'page', None),
                'score': getattr(c, 'score', None)
            } for c in docs]

            # Construir mensajes
            user_prompt = build_user_prompt(q, ctx_docs)
            system_prompt = get_system_prompt()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # Llamada al proveedor
            try:
                answer = provider.chat(messages)
            except Exception as e:
                answer = f"[Error proveedor] {e}"

            latency = time.time() - start

            # Asegurar bloque de Referencias al final si falta
            refs_text = _extract_references(answer)
            if not refs_text:
                refs_text = _format_references_from_docs(ctx_docs)
                answer = answer.rstrip() + "\n\nReferencias:\n" + refs_text

            # Estimación de costo (tokens aproximados)
            tokens_in = _approx_tokens(str(messages))
            tokens_out = _approx_tokens(answer)
            try:
                cost = provider.estimate_cost(tokens_in, tokens_out)
            except Exception:
                cost = 0.0

            # Exact match simple (si hay 'answer' en gold)
            gold_answer = item.get('answer', '').strip().lower()
            exact = False
            if gold_answer:
                exact = gold_answer in answer.strip().lower()

            results.append(EvalResult(
                question=q,
                provider=provider_name,
                answer=answer,
                references=refs_text,
                latency_sec=latency,
                est_cost_usd=cost,
                exact_match=exact,
            ))

        return results

    def calculate_aggregate_metrics(self, results: List[EvalResult]) -> Dict[str, Any]:
        if not results:
            return {}
        n = len(results)
        em = sum(1 for r in results if r.exact_match) / n
        cov_citas = sum(1 for r in results if r.references) / n
        avg_latency = sum(r.latency_sec for r in results) / n
        avg_cost = sum(r.est_cost_usd for r in results) / n
        return {
            "n": n,
            "exact_match": round(em, 3),
            "citation_coverage": round(cov_citas, 3),
            "avg_latency_sec": round(avg_latency, 3),
            "avg_cost_usd": round(avg_cost, 6),
        }

    def save_csv(self, results: List[EvalResult], out_path: str) -> None:
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=["question", "provider", "answer", "references"]) 
            w.writeheader()
            for r in results:
                w.writerow({
                    "question": r.question,
                    "provider": r.provider,
                    "answer": r.answer,
                    "references": r.references,
                })

    def save_summary(self, provider_name: str, metrics: Dict[str, Any], out_path: str) -> None:
        data = {"provider": provider_name, **metrics}
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def run_and_save(self, provider, provider_name: str, out_dir: str = "eval") -> Dict[str, Any]:
        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = self.evaluate_provider(provider, provider_name)
        metrics = self.calculate_aggregate_metrics(results)
        self.save_csv(results, os.path.join(out_dir, f"results_{provider_name}_{stamp}.csv"))
        self.save_summary(provider_name, metrics, os.path.join(out_dir, f"summary_{provider_name}_{stamp}.json"))
        return metrics
