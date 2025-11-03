import requests
import base64
import re
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st

# ------------------- Helper functions -------------------
def get_repo_info(url):
    """Extract owner and repo name from GitHub URL"""
    parts = url.strip().replace("https://github.com/", "").split("/")
    if len(parts) < 2:
        raise ValueError("Invalid GitHub URL format.")
    return parts[0], parts[1].replace(".git", "")

def fetch_repo_files(owner, repo, path=""):
    """Recursively fetch all files from a GitHub repository."""
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    response = requests.get(api_url)
    if response.status_code != 200:
        st.error(f"Failed to fetch: {api_url} ({response.status_code})")
        return []

    items = response.json()
    files = []

    for item in items:
        if item["type"] == "file":
            files.append(item)
        elif item["type"] == "dir":
            files.extend(fetch_repo_files(owner, repo, item["path"]))

    return files

def fetch_file_content(file_url):
    """Get file content (base64-decoded)."""
    res = requests.get(file_url)
    if res.status_code == 200:
        content = res.json().get("content", "")
        return base64.b64decode(content).decode("utf-8", errors="ignore")
    return ""

def extract_imports(code):
    """Extract import statements from Python code."""
    imports = []
    for line in code.splitlines():
        line = line.strip()
        if line.startswith("import "):
            imp = line.split("import ")[1].split(" as ")[0].strip()
            imports.append(imp)
        elif line.startswith("from "):
            imp = line.split("import")[0].replace("from", "").strip()
            imports.append(imp)
    return imports

def build_internal_dependencies(files):
    """Find relationships between files within the repo."""
    module_map = {f["path"][:-3].replace("/", "."): f["path"] for f in files if f["path"].endswith(".py")}
    dependencies = {}

    for file in files:
        if not file["path"].endswith(".py"):
            continue

        content = fetch_file_content(file["url"])
        imports = extract_imports(content)
        internal_imports = [i for i in imports if i in module_map]

        dependencies[file["path"]] = internal_imports
    return dependencies

def visualize_dependencies(dependencies):
    """Draw dependency graph."""
    G = nx.DiGraph()
    for f, deps in dependencies.items():
        G.add_node(f)
        for d in deps:
            G.add_edge(f, d)

    plt.figure(figsize=(14, 9))
    pos = nx.spring_layout(G, k=0.5)
    nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=2000, font_size=8, arrows=True)
    plt.title("📊 Internal File Dependency Graph", fontsize=16)
    st.pyplot(plt)

# ------------------- Streamlit UI -------------------
st.title("🧩 GitHub Repository Dependency Visualizer")
st.markdown("""
Analyze file-level dependencies and structure of any **public GitHub repository**  
without cloning it locally.
""")

github_url = st.text_input("🔗 Enter Public GitHub Repository URL:")

if github_url:
    try:
        owner, repo = get_repo_info(github_url)
        st.info(f"Fetching files from **{owner}/{repo}** ...")
        files = fetch_repo_files(owner, repo)

        py_files = [f for f in files if f["path"].endswith(".py")]
        st.success(f"✅ Found {len(py_files)} Python files.")

        dependencies = build_internal_dependencies(py_files)

        if dependencies:
            st.subheader("📁 Internal Dependencies Found:")
            for file, deps in dependencies.items():
                if deps:
                    st.markdown(f"**{file}** → {', '.join(deps)}")

            visualize_dependencies(dependencies)
        else:
            st.warning("No internal dependencies detected.")
    except Exception as e:
        st.error(f"Error: {e}")
