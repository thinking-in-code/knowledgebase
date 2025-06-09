import chunk
import chromadb
import embed

chromadb_client = chromadb.PersistentClient("./chroma.db")
chromadb_collection = chromadb_client.get_or_create_collection("ragtest")

def create_db() -> None:
    for idx, c in enumerate(chunk.get_chunks()):
        print(f"Process: {c}")
        embedding = embed.embed(c)
        chromadb_collection.upsert(
            ids=str(idx),
            documents=c,
            embeddings=embedding
        )


def query_db(question: str) -> list[str]:
    question_embedding = embed.embed(question)
    result = chromadb_collection.query(
        query_embeddings=question_embedding,
        n_results=3
    )
    return result["documents"][0]

if __name__ == '__main__':
    # 将向量化的切片存储到向量数据库
    create_db()
    # 向量库中查询
    question = "关羽为什么和红孩儿打架？"
    query_db(question)
