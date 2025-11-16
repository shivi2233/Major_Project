import os
import tempfile
from git import Repo
import networkx as nx
from pyvis.network import Network
import streamlit as st
from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain_core.runnables import RunnableBranch, RunnableParallel, RunnableLambda, RunnableSequence

# ─────────────────────────────────────────────
# 1️⃣ Load environment variables
# ─────────────────────────────────────────────
load_dotenv()
api_key = os.getenv("COHERE_API_KEY")

if not api_key:
    raise ValueError("❌ COHERE_API_KEY not found in .env file!")

# ─────────────────────────────────────────────
# 2️⃣ Initialize Cohere LLM
# ─────────────────────────────────────────────
llm = ChatCohere(cohere_api_key=api_key, model="command-a-03-2025", temperature=0.3)


# ─────────────────────────────────────────────
# 3️⃣ Core helper functions
# ─────────────────────────────────────────────
def detect_input_type(inputs):
    """Detect whether user uploaded a single file or provided a repo link."""
    inp = inputs.get("input_path", "")
    file_obj = inputs.get("file_obj")

    # Supported single-file extensions
    supported_exts = (".py", ".js", ".cpp", ".java", ".ipynb", ".csv", ".sh", ".json", ".html", ".md")

    if inp.endswith(supported_exts) or file_obj:
        return {"input_type": "file"}
    return {"input_type": "repo"}




def load_file(inputs):
    """Load file content."""
    file_obj = inputs.get("file_obj")
    if file_obj:
        content = file_obj.read().decode("utf-8")
    else:
        with open(inputs["input_path"], "r") as f:
            content = f.read()
    return {"file_content": content}


def analyze_file(inputs):
    """Analyze single file."""
    content = inputs["file_content"]
    prompt = f"""
    You are a code analysis assistant. Analyze the following code and provide:
    1. Summary
    2. Key concepts
    3. Optimization suggestions
    4. Dependencies
    Code:
    {content}
    """
    response = llm.invoke(prompt)
    return {"analysis": response.content}


def load_repo(inputs):
    """Clone GitHub repo."""
    repo_url = inputs["input_path"]
    tmp_dir = tempfile.mkdtemp()
    Repo.clone_from(repo_url, tmp_dir)
    return {"repo_path": tmp_dir}


def analyze_repo_structure(inputs):
    repo_path = inputs["repo_path"]
    structure = {"frontend": [], "backend": [], "db": [], "env": []}

    for root, _, files in os.walk(repo_path):
        for f in files:
            fp = os.path.join(root, f)
            if f.endswith((".js", ".jsx", ".ts", ".html", ".css")):
                structure["frontend"].append(fp)
            elif f.endswith((".py", ".java", ".cpp")):
                structure["backend"].append(fp)
            elif "env" in f or f.endswith(".env"):
                structure["env"].append(fp)
            elif f.endswith((".sql", ".db")):
                structure["db"].append(fp)

    return {**inputs, "repo_structure": structure}




def analyze_files(inputs):
    """Sequentially analyze selected files."""
    summaries = []
    for file in inputs["selected_files"]:
        with open(file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        prompt = f"You are a code reviewer. Analyze:\n{content}"
        response = llm.invoke(prompt)
        summaries.append(f"### {os.path.basename(file)}\n{response.content}\n")
    return {"summary": "\n".join(summaries)}




import os
import networkx as nx
from pyvis.network import Network

def build_dependency_graph(inputs):
    """
    Builds a dependency graph from selected repo files or selected files list.
    If no valid files are found, generates a dummy graph with a single node.
    Works safely even when pyvis template rendering fails.
    """
    repo_path = inputs.get("repo_path")
    selected_files = inputs.get("selected_files", [])

    valid_exts = (".py", ".js", ".cpp", ".java", ".ipynb", ".sh", ".csv")
    files_to_scan = []

    if selected_files:
        for f in selected_files:
            if f.endswith(valid_exts):
                files_to_scan.append(os.path.join(repo_path, f) if repo_path else f)
    elif repo_path:
        for root, _, files in os.walk(repo_path):
            for f in files:
                if f.endswith(valid_exts):
                    files_to_scan.append(os.path.join(root, f))

    # Create graph
    G = nx.DiGraph()

    # ✅ Handle case when no valid files are found
    if not files_to_scan:
        G.add_node("No valid files found")

    else:
        for f in files_to_scan:
            G.add_node(f)
            try:
                with open(f, "r", encoding="utf-8") as file:
                    for line in file:
                        if "import" in line or "#include" in line:
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                dep = parts[1]
                                G.add_edge(f, dep)
            except Exception as e:
                print(f"Error reading {f}: {e}")

    # ✅ Safely render the graph HTML
    graph_path = os.path.join("graph_output.html")
    try:
        net = Network(height="450px", width="100%", bgcolor="#ffffff", directed=True)

        for node in G.nodes():
            net.add_node(node, label=os.path.basename(node))

        for src, dst in G.edges():
            net.add_edge(src, dst)

        # Try to show using pyvis’ internal template
        try:
            net.show(graph_path)
        except Exception as e:
            print(f"Pyvis rendering failed: {e}")
            # Fallback: manually write a basic HTML file
            net.save_graph(graph_path)

    except Exception as e:
        print(f"Graph generation failed: {e}")
        with open("graph_output.html", "w", encoding="utf-8") as f:
            f.write("<h3>Unable to render dependency graph.</h3>")

    return {
        "graph_path": graph_path,
        "summary": f"Dependency graph generated with {len(G.nodes())} nodes and {len(G.edges())} edges."
    }


def summarize_repo(inputs):
    prompt = f"""
    Summarize the repository based on these analyses:
    {inputs['summary']}
    Provide overall design patterns and improvements.
    """
    response = llm.invoke(prompt)
    return {"final_summary": response.content, "graph_path": inputs["graph_path"]}


# ─────────────────────────────────────────────
# 4️⃣ Define LangChain flows
# ─────────────────────────────────────────────
# ✅ Step 1: Make sure your file_flow, repo_flow, and fallback_chain are defined above

# File flow
# ✅ Step 1: define your flows first (same as before)
file_flow = RunnableSequence(
    RunnableLambda(load_file),
    RunnableLambda(analyze_file)
)

repo_flow = RunnableSequence(
    RunnableLambda(load_repo),
    RunnableLambda(analyze_repo_structure),
    RunnableLambda(analyze_files),
    RunnableLambda(build_dependency_graph),
    RunnableLambda(summarize_repo)
)



fallback_chain = RunnableLambda(lambda _: {"error": "Unsupported input type"})

input_detector = RunnableLambda(detect_input_type)

# ✅ Step 2: RunnableBranch for langchain_core 1.0.3
def route_input(inputs):
    print("DEBUG INPUTS:", inputs)   # ← Add this
    if "file_obj" in inputs:
        return file_flow.invoke(inputs)
    elif "input_path" in inputs:
        return repo_flow.invoke(inputs)
    else:
        raise ValueError("Invalid input. Provide file_obj or input_path.")





branch_flow = RunnableLambda(route_input)
main_flow = branch_flow







# ─────────────────────────────────────────────
# 5️⃣ Streamlit UI
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────
st.set_page_config(page_title="Code Context Reviewer", layout="wide")
st.title("🧠 Code Context Reviewer (Cohere + LangChain)")

option = st.radio("Choose Input Type", ["Single File", "Repository Link"])

if option == "Single File":
    file = st.file_uploader("Upload file", type=["py", "js", "cpp", "java"])
    if file and st.button("Analyze File"):
        result = main_flow.invoke({"file_obj": file})
        st.markdown(result["analysis"])

else:
    import re
    import json
    import math
    import requests
    import tempfile
    from urllib.parse import urlparse

    st.subheader("📁 Analyze a GitHub Repository (no clone, token optional)")

    # Use GITHUB_TOKEN from .env if available (recommended for private repos / higher rate limit)
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # set this in .env if you have a PAT

    # ----------------- Helper: GitHub API & raw fetch -----------------
    def github_api_get(url, params=None):
        headers = {"Accept": "application/vnd.github.v3+json"}
        if GITHUB_TOKEN:
            headers["Authorization"] = f"token {GITHUB_TOKEN}"
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def parse_github_repo(url: str):
        """Return (owner, repo) or None if not a GitHub repo."""
        parsed = urlparse(url)
        if "github.com" not in parsed.netloc:
            return None
        parts = parsed.path.strip("/").split("/")
        if len(parts) < 2:
            return None
        owner, repo = parts[0], parts[1].replace(".git", "")
        return owner, repo

    def fetch_repo_file_list(owner: str, repo: str):
        """
        Try to fetch a recursive file list using the trees API. Fallback to recursive walk on contents API.
        Returns list of {path, size, raw_url}
        """
        files = []
        try:
            repo_meta = github_api_get(f"https://api.github.com/repos/{owner}/{repo}")
            default_branch = repo_meta.get("default_branch", "main")
            tree_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{default_branch}"
            tree = github_api_get(tree_url, params={"recursive": "1"})
            for item in tree.get("tree", []):
                if item.get("type") == "blob":
                    path = item["path"]
                    size = item.get("size", 0)
                    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{default_branch}/{path}"
                    files.append({"path": path, "size": size, "raw_url": raw_url})
            return files
        except Exception:
            # Fallback: contents API recursive walk
            try:
                def walk_contents(api_url, base=""):
                    entries = github_api_get(api_url)
                    for ent in entries:
                        if ent["type"] == "file":
                            files.append({
                                "path": os.path.join(base, ent["name"]),
                                "size": ent.get("size", 0),
                                "raw_url": ent.get("download_url")
                            })
                        elif ent["type"] == "dir":
                            walk_contents(ent["url"], os.path.join(base, ent["name"]))
                walk_contents(f"https://api.github.com/repos/{owner}/{repo}/contents")
                return files
            except Exception as e:
                raise RuntimeError(f"Failed to list repo files: {e}")

    def fetch_file_preview(raw_url: str, max_chars=2000):
        """Fetch a preview snippet for a file; returns empty string on failure."""
        if not raw_url:
            return ""
        try:
            headers = {}
            if GITHUB_TOKEN:
                headers["Authorization"] = f"token {GITHUB_TOKEN}"
            r = requests.get(raw_url, headers=headers, timeout=20)
            r.raise_for_status()
            text = r.text
            # Truncate long files for previews
            return text[:max_chars]
        except Exception:
            return ""

    def fetch_file_full(raw_url: str, max_chars=500000):
        """Fetch larger file content for selected files (safe fallback)."""
        if not raw_url:
            return ""
        try:
            headers = {}
            if GITHUB_TOKEN:
                headers["Authorization"] = f"token {GITHUB_TOKEN}"
            r = requests.get(raw_url, headers=headers, timeout=40)
            r.raise_for_status()
            return r.text[:max_chars]
        except Exception:
            return ""

    # ----------------- LLM: categorize & metadata extraction -----------------
    def llm_categorize_files(file_infos):
        """
        Ask the LLM to create dynamic categories and produce metadata for each file.
        file_infos: list of {path, size, raw_url, preview}
        Returns parsed JSON {categories: {...}, file_metadata: {...}}
        """
        # build small payload
        sample = []
        for f in file_infos:
            sample.append({"path": f["path"], "size": f["size"], "preview": (f["preview"][:600] + " ...") if f["preview"] else ""})

        prompt = f"""
You are an expert codebase analyst. I will give you a JSON array of files (path, size, preview).
Tasks:
1) Create dynamic categories for this repo (e.g., frontend, backend, scripts, data, docs, tests, infra, env, notebooks, config).
2) Assign each file to one category.
3) For each file provide inferred_type (language/format) and a 1-line description.
Output strictly valid JSON with two keys: "categories" and "file_metadata".
"categories" maps category -> list of file paths.
"file_metadata" maps path -> {{size, inferred_type, description}}.

Files:
{json.dumps(sample, ensure_ascii=False)}
"""
        resp = llm.invoke(prompt)
        txt = resp.content.strip()
        try:
            parsed = json.loads(txt)
        except Exception:
            m = re.search(r"(\{[\s\S]*\})", txt)
            if m:
                parsed = json.loads(m.group(1))
            else:
                # fallback heuristic
                parsed = {"categories": {}, "file_metadata": {}}
                for f in file_infos:
                    ext = os.path.splitext(f["path"])[1].lower()
                    cat = {
                        ".py": "backend",
                        ".ipynb": "notebooks",
                        ".js": "frontend",
                        ".html": "frontend",
                        ".css": "frontend",
                        ".md": "docs",
                        ".csv": "data",
                        ".sh": "scripts",
                        ".json": "config"
                    }.get(ext, "other")
                    parsed.setdefault("categories", {}).setdefault(cat, []).append(f["path"])
                    parsed.setdefault("file_metadata", {})[f["path"]] = {
                        "size": f["size"],
                        "inferred_type": ext.lstrip(".") or "unknown",
                        "description": f"Auto-inferred from extension ({ext})"
                    }

        # ensure metadata fields
        for f in file_infos:
            meta = parsed.setdefault("file_metadata", {}).setdefault(f["path"], {})
            meta.setdefault("size", f.get("size", 0))
            meta.setdefault("preview", f.get("preview", "")[:1200])
            meta.setdefault("inferred_type", meta.get("inferred_type") or os.path.splitext(f["path"])[1].lstrip(".") or "unknown")

        return parsed

    # ----------------- LLM: analyze selected files in batches with progress -----------------
    def llm_analyze_files_batch(repo_context, file_paths, file_full_contents_map, batch_size=4):
        """
        Analyze files in batches. Returns mapping path -> analysis dict.
        Shows progress using st.progress.
        """
        total = len(file_paths)
        analyses = {}
        if total == 0:
            return analyses

        progress_bar = st.progress(0)
        batches = [file_paths[i:i + batch_size] for i in range(0, total, batch_size)]
        processed = 0

        for bi, batch in enumerate(batches):
            # build prompt for this batch
            file_entries = []
            for p in batch:
                file_entries.append({"path": p, "content": file_full_contents_map.get(p, "")[:6000]})

            prompt = f"""
You are a senior engineer. Repo context (categories & metadata):
{json.dumps(repo_context, ensure_ascii=False)}

For each of the following files, return a JSON array of objects with:
- path
- summary (1-2 sentences)
- dependencies (list)
- suggestions (list of short suggestions)

Files:
{json.dumps(file_entries, ensure_ascii=False)}
"""
            resp = llm.invoke(prompt)
            txt = resp.content.strip()
            try:
                parsed = json.loads(txt)
            except Exception:
                m = re.search(r"(\[.*\])", txt, re.S)
                if m:
                    parsed = json.loads(m.group(1))
                else:
                    # fallback minimal outputs
                    parsed = []
                    for p in batch:
                        parsed.append({"path": p, "summary": "No analysis", "dependencies": [], "suggestions": []})

            for item in parsed:
                analyses[item["path"]] = item

            processed += len(batch)
            progress_bar.progress(min(1.0, processed / total))
            # small sleep could be added to avoid rate bursts, but not necessary

        progress_bar.empty()
        return analyses

    # ----------------- UI: load repo metadata (no clone) -----------------
    repo_url = st.text_input("Enter GitHub Repo URL:")

    if st.button("🔍 Load Repository (no clone)"):
        if not repo_url:
            st.error("Please enter a GitHub repository URL.")
        else:
            owner_repo = parse_github_repo(repo_url)
            if not owner_repo:
                st.error("Only GitHub repository URLs are supported for no-clone mode.")
            else:
                owner, repo = owner_repo
                with st.spinner("Fetching file list from GitHub..."):
                    try:
                        files = fetch_repo_file_list(owner, repo)
                    except Exception as e:
                        st.error(f"Failed to fetch repo file list: {e}")
                        files = []

                # fetch previews (limit to first N to reduce requests); keep rest empty
                MAX_PREVIEWS = 120  # adjust as needed (number of files to prefetch)
                for i, f in enumerate(files):
                    if i < MAX_PREVIEWS:
                        f["preview"] = fetch_file_preview(f["raw_url"], max_chars=2000)
                    else:
                        f["preview"] = ""

                with st.spinner("Asking LLM to dynamically categorize files..."):
                    categorized = llm_categorize_files(files)

                st.session_state["repo_data"] = {
                    "owner": owner,
                    "repo": repo,
                    "files": files,
                    "categorized": categorized
                }
                st.success("Repository metadata fetched and categorized (no clone).")

    # ----------------- UI: show categories & select files -----------------
    if "repo_data" in st.session_state:
        repo_data = st.session_state["repo_data"]
        categorized = repo_data.get("categorized", {})
        categories = categorized.get("categories", {})
        file_metadata = categorized.get("file_metadata", {})
        files_list = repo_data.get("files", [])

        st.write("### Dynamic Categories")
        cols = st.columns(3)
        idx = 0
        for cat, paths in categories.items():
            with cols[idx % 3]:
                st.write(f"**{cat}** — {len(paths)} files")
            idx += 1

        chosen_category = st.selectbox("Choose category to inspect", ["All"] + list(categories.keys()))
        if chosen_category == "All":
            visible_files = [p for arr in categories.values() for p in arr]
        else:
            visible_files = categories.get(chosen_category, [])

        # show file listing with small metadata and "View raw" button
        st.markdown("#### Files (click View to fetch raw contents)")
        for path in visible_files:
            meta = file_metadata.get(path, {})
            cols = st.columns([6, 1, 1])
            cols[0].markdown(f"- `{path}` — {meta.get('inferred_type', '')} — {meta.get('description','')}")
            if cols[1].button("View raw", key=f"view_{path}"):
                # fetch raw content and show in an expander
                info = next((f for f in files_list if f["path"] == path), None)
                if info:
                    full = fetch_file_full(info["raw_url"], max_chars=200000)
                    st.expander(f"Raw: {path}", expanded=True).code(full if full else "<empty or binary file>")
            # small spacer column for layout; we don't need cols[2] right now

        # multi-select for analysis
        selected_files = st.multiselect("Select files to analyze (multi-select)", visible_files, default=visible_files if visible_files else None)
        analyze_all_flag = st.checkbox("Analyze all categorized files (ignore selection)", value=False)

        if st.button("🚀 Run Analysis (LLM-powered)"):
            if analyze_all_flag:
                chosen_paths = [p for arr in categories.values() for p in arr]
            else:
                chosen_paths = selected_files or [p for arr in categories.values() for p in arr]

            if not chosen_paths:
                st.error("No files selected and repository appears empty.")
            else:
                with st.spinner("Fetching selected file contents and running LLM analysis..."):
                    raw_map = {f["path"]: f for f in files_list}
                    full_contents = {}
                    # Fetch in batches to avoid too many requests at once
                    for p in chosen_paths:
                        info = raw_map.get(p)
                        full_contents[p] = fetch_file_full(info["raw_url"]) if info else ""

                    repo_context = {"categories": categories, "file_metadata": file_metadata, "owner": repo_data["owner"], "repo": repo_data["repo"]}

                    # run analysis in batches with a progress bar
                    analyses_map = llm_analyze_files_batch(repo_context, chosen_paths, full_contents, batch_size=3)

                    st.session_state["analysis_result"] = {
                        "repo_context": repo_context,
                        "analyses_map": analyses_map,
                        "owner": repo_data["owner"],
                        "repo": repo_data["repo"]
                    }
                    st.success("✅ LLM analysis complete. Scroll down for results.")

    # ----------------- Display results & dependency graph -----------------
    if "analysis_result" in st.session_state:
        res = st.session_state["analysis_result"]
        st.markdown("## 🧾 Per-file Analysis (LLM)")
        for path, analysis in res["analyses_map"].items():
            st.markdown(f"### `{path}`")
            st.markdown(f"**Summary:** {analysis.get('summary','-')}")
            st.markdown(f"**Dependencies:** {analysis.get('dependencies',[])}")
            st.markdown("**Suggestions:**")
            for s in analysis.get("suggestions", []):
                st.markdown(f"- {s}")

        # simple dependency graph from LLM dependencies
        G = nx.DiGraph()
        for p in res["analyses_map"]:
            G.add_node(p, label=os.path.basename(p))
        basenames = {p: os.path.basename(p) for p in res["analyses_map"].keys()}
        for p, analysis in res["analyses_map"].items():
            deps = analysis.get("dependencies", [])
            for dep in deps:
                for other_p, base in basenames.items():
                    if base in str(dep) and other_p != p:
                        G.add_edge(p, other_p)

        # render pyvis graph safely (with color coding)
        graph_path = os.path.join(tempfile.gettempdir(), f"repo_graph_{res['owner']}_{res['repo']}.html")
        net = Network(height="600px", width="100%", bgcolor="#ffffff", directed=True)
        for node in G.nodes():
            ext = os.path.splitext(node)[1].lower()
            color = "#97C2FC"
            if ext in (".py",):
                color = "#ffcc00"
            elif ext in (".js", ".ts"):
                color = "#66cc66"
            elif ext in (".ipynb",):
                color = "#ff99cc"
            elif ext in (".csv", ".json"):
                color = "#ff6666"
            net.add_node(node, label=os.path.basename(node), color=color)

        for src, dst in G.edges():
            net.add_edge(src, dst)

        try:
            net.show(graph_path)
        except Exception:
            try:
                net.save_graph(graph_path)
            except Exception:
                # last-resort write minimal html
                with open(graph_path, "w", encoding="utf-8") as fh:
                    fh.write("<h3>Unable to render interactive graph.</h3>")

        if os.path.exists(graph_path):
            st.markdown("### 🔗 Dependency Graph (interactive)")
            st.components.v1.html(open(graph_path, "r", encoding="utf-8").read(), height=700)
        else:
            st.warning("Could not produce dependency graph file.")
