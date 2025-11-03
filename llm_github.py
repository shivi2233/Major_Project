import requests

def fetch_repo_structure(owner, repo, branch="main"):
    """Fetch all file paths and raw URLs from a GitHub repo"""
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    res = requests.get(url)
    res.raise_for_status()
    data = res.json()

    files = []
    for item in data.get("tree", []):
        if item["type"] == "blob":  # only files
            files.append({
                "path": item["path"],
                "url": f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{item['path']}"
            })
    return files
def fetch_file_contents(files, size_limit_kb=30):
    """Fetch the contents of all text/code files in the repo"""
    repo_data = []
    for file in files:
        try:
            r = requests.get(file["url"], timeout=10)
            r.raise_for_status()
            text = r.text

            # Skip binary or oversized files
            if len(text.encode('utf-8')) > size_limit_kb * 1024:
                continue

            repo_data.append({
                "filename": file["path"],
                "content": text
            })
        except Exception as e:
            print(f"Skipping {file['path']}: {e}")
    return repo_data
import json

owner = "streamlit"
repo = "streamlit"
branch = "develop"

files = fetch_repo_structure(owner, repo, branch)
repo_data = fetch_file_contents(files)

with open("repo_data.json", "w", encoding="utf-8") as f:
    json.dump(repo_data, f, indent=2, ensure_ascii=False)
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
import json

# Load JSON data
with open("repo_data.json", "r", encoding="utf-8") as f:
    repo_json = f.read()

# Create LLM pipeline
llm = ChatOpenAI(model="gpt-4o", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""
You are a multi-language code dependency analyzer.
Given a repository's files and their contents, analyze:
1. How files depend on each other (imports, calls, includes, links)
2. Which are frontend, backend, config, test, or utility
3. Output a structured JSON showing relationships and roles
"""),
    HumanMessage(content="Repository data:\n{repo_json}")
])

chain = prompt | llm
response = chain.invoke({"repo_json": repo_json})

print(response.content)
import networkx as nx
import matplotlib.pyplot as plt
import json

with open("llm_output.json", "r") as f:
    data = json.load(f)

G = nx.DiGraph()
for dep in data["dependencies"]:
    G.add_edge(dep["from"], dep["to"])

plt.figure(figsize=(10, 8))
nx.draw(G, with_labels=True, node_color="skyblue", node_size=2000, arrows=True)
plt.show()
