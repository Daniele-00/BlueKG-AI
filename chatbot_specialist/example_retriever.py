import yaml
import pickle
from pathlib import Path
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)


DEFAULT_TOP_K = int(os.getenv("EXAMPLES_TOP_K", "3"))


class ExampleRetriever:
    def __init__(
        self,
        examples_dir: str = "examples",
        cache_dir: str = "embeddings",
        default_top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
    ):
        self.examples_dir = Path(examples_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.default_top_k = default_top_k or DEFAULT_TOP_K
        self.min_similarity = (
            float(min_similarity) if min_similarity is not None else None
        )
        self.last_similarity: Optional[float] = None

        logger.info("Caricamento modello embeddings...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.examples_by_specialist = {}
        self._load_all_examples()

    def _load_all_examples(self):
        specialists = [
            "SALES_CYCLE",
            "PURCHASE_CYCLE",
            "GENERAL_QUERY",
            "DITTE_QUERY",
            "CROSS_DOMAIN",
        ]

        for specialist in specialists:
            yaml_path = self.examples_dir / f"{specialist}.yaml"
            cache_path = self.cache_dir / f"{specialist}_vectors.pkl"

            if not yaml_path.exists():
                logger.warning(f"{yaml_path} mancante")
                self.examples_by_specialist[specialist] = []
                continue

            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                examples = data.get("examples", [])

            if not examples:
                self.examples_by_specialist[specialist] = []
                continue

            # Cache embeddings
            if cache_path.exists():
                logger.info(f"Cache {specialist}")
                with open(cache_path, "rb") as f:
                    embeddings = pickle.load(f)
            else:
                logger.info(f"Embedding {specialist} ({len(examples)} esempi)")
                questions = [ex["question"] for ex in examples]
                embeddings = self.model.encode(questions, show_progress_bar=False)

                with open(cache_path, "wb") as f:
                    pickle.dump(embeddings, f)

            for ex, emb in zip(examples, embeddings):
                ex["embedding"] = emb

            self.examples_by_specialist[specialist] = examples
            logger.info(f"{specialist}: {len(examples)} esempi")

    def retrieve(
        self,
        question: str,
        specialist: str,
        top_k: Optional[int] = None,
        allow_low_similarity: bool = False,
    ) -> List[Dict]:
        examples = self.examples_by_specialist.get(specialist, [])
        if not examples:
            return []

        k = top_k if top_k is not None else self.default_top_k
        if k <= 0:
            return []

        q_embedding = self.model.encode([question])[0]
        q_norm = np.linalg.norm(q_embedding)
        if q_norm == 0:
            logger.warning(
                "Embedding domanda con norma nulla; nessun esempio restituito."
            )
            self.last_similarity = None
            return []

        similarities = []
        for ex in examples:
            emb = ex.get("embedding")
            if emb is None:
                continue
            emb_norm = np.linalg.norm(emb)
            if emb_norm == 0:
                continue
            sim = np.dot(q_embedding, emb) / (q_norm * emb_norm)
            similarities.append((sim, ex))

        similarities.sort(key=lambda x: x[0], reverse=True)
        if not similarities:
            self.last_similarity = None
            return []

        best_score = float(similarities[0][0])
        self.last_similarity = best_score

        if (
            not allow_low_similarity
            and self.min_similarity is not None
            and best_score < self.min_similarity
        ):
            logger.info(
                "Soglia esempi non raggiunta (%.3f < %.3f) per specialist %s. Nessun esempio usato.",
                best_score,
                self.min_similarity,
                specialist,
            )
            return []

        selected = []
        for score, ex in similarities[:k]:
            item = dict(ex)
            item.pop("embedding", None)
            item["similarity"] = float(score)
            selected.append(item)
        return selected

    def get_stats(self) -> Dict:
        return {
            specialist: len(examples)
            for specialist, examples in self.examples_by_specialist.items()
        }

    def get_default_top_k(self) -> int:
        return self.default_top_k
