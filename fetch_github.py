import streamlit as st
import requests
import json
import os
import cohere
from dotenv import load_dotenv
import networkx as nx
import matplotlib.pyplot as plt

# ---------------------- SETUP ----------------------
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

if not COHERE_API_KEY:
    st.error("⚠️ Cohere API key not found! Please add it to your `.env` file as COHERE_API_KEY.")
    st.stop()

co = cohere.ClientV2(api_key=COHERE_API_KEY)

# ---------------------- GITHUB FUNCTIONS ----------------------
def fetch_repo_structure(owner, repo, branch="main"):
    """Fetch all file paths and raw URLs from a GitHub repo."""
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    res = requests.get(url)
    if res.status_code != 200:
        raise Exception(f"Failed to fetch repo structure: {res.status_code}")
    data = res.json()
    files = []
    for item in data.get("tree", []):
        if item["type"] == "blob":
            files.append({
                "path": item["path"],
                "url": f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{item['path']}"
            })
    return files

def fetch_file_contents(files, size_limit_kb=30, max_files=50):
    """Fetch the contents of limited number of text/code files."""
    repo_data = []
    for file in files[:max_files]:
        try:
            r = requests.get(file["url"], timeout=10)
            r.raise_for_status()
            text = r.text
            if len(text.encode("utf-8")) > size_limit_kb * 1024:
                continue
            repo_data.append({
                "filename": file["path"],
                "content": text
            })
        except Exception:
            continue
    return repo_data

# ---------------------- COHERE ANALYSIS ----------------------
def analyze_dependencies_with_cohere(repo_data):
    """Use Cohere Chat API (V2) to analyze dependencies and file roles."""
    repo_json = json.dumps(repo_data, ensure_ascii=False)

    prompt = f"""
You are a multi-language code dependency analyzer.
Given a repository’s structure and content:

1. Identify dependencies between files (imports, includes, links, or usage).
2. Categorize each file as one of: frontend, backend, config, test, utils, documentation.
3. Return structured JSON only in the following format:

{{
  "dependencies": [{{"from": "fileA", "to": "fileB"}}],
  "roles": [{{"file": "filename", "role": "role"}}]
}}

Repository data:
{repo_json}
"""

    try:
        response = co.chat(
            model="command-r-plus",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        output_text = response.message.content[0].text
        try:
            parsed = json.loads(output_text)
        except json.JSONDecodeError:
            parsed = {"raw_output": output_text}

        return parsed

    except Exception as e:
        return {"error": str(e)}

# ---------------------- STREAMLIT UI ----------------------
st.title("📂 GitHub Repository Dependency Analyzer (Cohere + Langchain Style)")
st.markdown("Analyze **any GitHub repository’s file structure, dependencies, and roles** automatically!")

repo_url = st.text_input("Enter a GitHub Repository URL (e.g. https://github.com/streamlit/streamlit):")

if st.button("Analyze Repository"):
    if not repo_url.strip():
        st.error("Please enter a valid GitHub repository URL.")
    else:
        try:
            owner, repo = repo_url.strip().replace("https://github.com/", "").split("/")[:2]
            st.info(f"Fetching structure of `{owner}/{repo}`...")

            files = fetch_repo_structure(owner, repo)
            st.write(f"✅ Found `{len(files)}` files. Fetching up to 50 for analysis...")

            repo_data = fetch_file_contents(files)
            st.success(f"Fetched `{len(repo_data)}` files for analysis!")

            st.info("Analyzing code relationships with Cohere... please wait ⏳")
            analysis = analyze_dependencies_with_cohere(repo_data)

            if "error" in analysis:
                st.error(f"❌ Error: {analysis['error']}")
            else:
                st.subheader("📊 Dependency Analysis Result")
                st.json(analysis)

                if "dependencies" in analysis and isinstance(analysis["dependencies"], list):
                    G = nx.DiGraph()
                    for dep in analysis["dependencies"]:
                        if "from" in dep and "to" in dep:
                            G.add_edge(dep["from"], dep["to"])

                    plt.figure(figsize=(10, 8))
                    nx.draw(G, with_labels=True, node_color="lightblue", node_size=2000, arrows=True)
                    st.pyplot(plt)

        except Exception as e:
            st.error(f"⚠️ {e}")
