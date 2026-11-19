"""Answer questions by quoting passages from local documents."""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

NO_ANSWER = "That information is not in the documents."
WORD = re.compile(r"\b[a-z0-9]+\b")


def terms(text: str) -> list[str]:
    """Normalize case and common plural forms for lexical retrieval."""
    return [
        word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word
        for word in WORD.findall(text.lower())
        if word not in ENGLISH_STOP_WORDS and word not in {"do", "does", "did"}
    ]


@dataclass(frozen=True)
class Passage:
    source: str
    number: int
    text: str

    @property
    def citation(self) -> str:
        return f"[{self.source}, passage {self.number}]"


class DocumentQA:
    def __init__(self, folder: Path):
        if not folder.is_dir():
            raise ValueError(f"Document folder does not exist: {folder}")

        self.passages = []
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".txt", ".md"}:
                continue
            paragraphs = re.split(r"\n\s*\n", path.read_text(encoding="utf-8-sig"))
            number = 0
            for paragraph in paragraphs:
                # Bound long paragraphs using overlapping 80-word windows.
                words = paragraph.split()
                for start in range(0, len(words), 65):
                    chunk = " ".join(words[start:start + 80])
                    number += 1
                    self.passages.append(Passage(path.relative_to(folder).as_posix(), number, chunk))
                    if start + 80 >= len(words):
                        break

        if not any(terms(passage.text) for passage in self.passages):
            raise ValueError("Add a non-empty UTF-8 .txt or .md file to the document folder.")
        self.vectorizer = TfidfVectorizer(analyzer=terms)
        self.embeddings = self.vectorizer.fit_transform(p.text for p in self.passages)

    def retrieve(self, question: str) -> list[tuple[Passage, float]]:
        # TF-IDF vectors are L2-normalized, so their dot product is cosine similarity.
        query = self.vectorizer.transform([question])
        scores = (self.embeddings @ query.T).toarray().ravel()
        ranked = sorted(range(len(scores)), key=lambda index: -scores[index])
        return [(self.passages[index], float(scores[index]))
                for index in ranked[:3] if scores[index] > 0]

    def answer(self, question: str) -> tuple[str, list[tuple[Passage, float]]]:
        hits = self.retrieve(question)
        query_terms = set(terms(question))
        for passage, score in hits:
            if score < 0.20 or not query_terms:
                continue
            for sentence in re.split(r"(?<=[.!?])\s+", passage.text):
                evidence_terms = set(terms(sentence))
                if query_terms < evidence_terms:
                    return f"{sentence} {passage.citation}", hits
        return NO_ANSWER, hits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Question to ask the documents")
    parser.add_argument("--docs", type=Path, default=Path(__file__).resolve().parent / "docs",
                        help="Folder containing UTF-8 .txt or .md files")
    args = parser.parse_args()
    if not args.question.strip():
        parser.error("Enter a non-empty question.")
    try:
        answer, hits = DocumentQA(args.docs).answer(args.question)
    except (ValueError, OSError) as error:
        parser.exit(2, f"{error}\n")

    print(f"Answer\n{answer}")
    print("\nRetrieved passages")
    if not hits:
        print("No matching passages.")
    for passage, score in hits:
        print(f"{passage.citation} (similarity {score:.2f})\n{passage.text}\n")


if __name__ == "__main__":
    main()
