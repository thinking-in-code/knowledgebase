from openai import OpenAI
import os

aiClient = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def embed(text: str) -> list[float]:
    result = aiClient.embeddings.create(
        model="text-embedding-v4",
        input=text,
        dimensions=1024, # 指定向量维度（仅 text-embedding-v3及 text-embedding-v4支持该参数）
        encoding_format="float"
    )
    return result.data[0].embedding

if __name__ == '__main__':
    embed_result = embed('你好')
    print('-------------向量化后的结果---------------')
    print(embed_result)