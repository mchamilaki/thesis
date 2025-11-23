from datasets import load_dataset
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def main():
    print("Loading ubuntu_dialogue_qa dataset...")
    ds = load_dataset(
    "sedthh/ubuntu_dialogue_qa",
    split="train",
    download_mode="force_redownload",
    verification_mode="no_checks",
)


    # Use only a subset while testing (optional but recommended)
    ds = ds.select(range(2000))

    print(f"Dataset loaded: {len(ds)} samples")

    # 1. Convert dataset rows into text strings and metadata
    texts = []
    metadatas = []

    for row in ds:
        q = row["INSTRUCTION"]
        a = row["RESPONSE"]
        combined = f"Q: {q}\nA: {a}"
        texts.append(combined)

        metadatas.append(
            {
                "SOURCE": row.get("SOURCE", ""),
                "METADATA": row.get("METADATA", "")
            }
        )

    print("Creating embeddings model (this may take a moment)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Building FAISS index...")
    vectorstore = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)

    print("Saving FAISS index to ./ubuntu_qa_index ...")
    vectorstore.save_local("ubuntu_qa_index")

    print("✅ Done! FAISS index saved.")


if __name__ == "__main__":
    main()
