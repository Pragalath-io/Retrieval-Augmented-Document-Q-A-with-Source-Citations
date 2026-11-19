# Document Q&A with source citations

This Python CLI answers questions by retrieving passages from local documents and quoting the sentence that supports the answer. The bundled campus library and lab documents are fictional examples.

## Requirements

| Component | Requirement |
|---|---|
| Runtime | Python 3.10 or newer |
| Dependency | scikit-learn, installed with requirements.txt |
| Documents | A folder of UTF-8 .txt or .md files. Two examples are included. |
| Embeddings | Sparse TF-IDF vectors for document passages and questions |
| Retrieval | Cosine similarity, up to three matching passages |
| Answer | Extracted sentence with a filename and passage number, or an abstention |
| Runtime access | Local CPU. No API key or internet connection is needed once dependencies are installed. |

## Run

Open a terminal in the project folder.

```bash
python -m pip install -r requirements.txt
python app.py "When does the campus library open on weekdays?"
python app.py "What is the library membership fee?"
```

The first command that calls the app quotes the library opening time and cites `[library.txt, passage 1]`. The second prints `That information is not in the documents.` Both show any retrieved passages below the answer. A retrieved passage may be related without providing enough evidence to answer.

Use your own documents by passing a folder path.

```bash
python app.py --docs ./my_documents "Your question here"
```

## Design decisions

1. Split documents at blank lines. Split long paragraphs into windows of up to 80 words with a 15-word overlap. Keep each window's relative filename and passage number.
2. Fit TF-IDF on the passages, embed the question with the same vectorizer, and rank matches using cosine similarity.
3. Return a source sentence only if its passage meets a 0.20 similarity threshold and the sentence contains all normalized question terms plus additional terms. Otherwise, abstain. Display up to three positive-scoring passages for inspection.

I chose extractive answers so every returned sentence can be checked directly against its source. TF-IDF measures shared vocabulary. The evidence rule is a heuristic and can miss paraphrases, return a merely related sentence, or miss evidence spread across sentences. The abstention means the script did not find sufficient evidence under its rule.

## Submission

To publish, create a public GitHub repository and upload app.py, requirements.txt, README.md, and the docs folder. Submit the repository URL if your assignment portal asks for it.
