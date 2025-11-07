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
llm = ChatCohere(cohere_api_key=api_key, model="command-r", temperature=0.3)

# ─────────────────────────────────────────────
# 3️⃣ Core helper functions
# ─────────────────────────────────────────────
def detect_input_type(inputs):
    """Detect whether user uploaded a single file or repo link."""
    inp = inputs.get("input_path", "")
    if inp.endswith((".py", ".js", ".cpp", ".java")) or inputs.get("file_obj"):
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
    """Detect file categories in repo."""
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

    return {"repo_structure": structure}


def get_user_choice(inputs):
    repo_structure = inputs["repo_structure"]
    all_files = sum(repo_structure.values(), [])
    choice = st.radio("Choose analysis mode:", ("Analyze all", "Select specific"))
    if choice == "Select specific":
        selected = st.selectbox("Select file:", all_files)
        return {"user_choice": "specific", "selected_files": [selected]}
    else:
        return {"user_choice": "all", "selected_files": all_files}


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


def build_dependency_graph(inputs):
    G = nx.DiGraph()
    for f in inputs["selected_files"]:
        G.add_node(os.path.basename(f))
        G.add_edge("Repo", os.path.basename(f))

    net = Network(height="400px", width="100%", bgcolor="#222222", font_color="white")
    net.from_nx(G)
    html_path = os.path.join(tempfile.gettempdir(), "dependency_graph.html")
    net.show(html_path)
    return {"graph_path": html_path, "summary": inputs["summary"]}


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

repo_parallel = RunnableParallel({
    "repo_structure": RunnableLambda(analyze_repo_structure),
    "user_choice": RunnableLambda(get_user_choice)
})

repo_flow = RunnableSequence(
    RunnableLambda(load_repo),
    repo_parallel,
    RunnableLambda(analyze_files),
    RunnableLambda(build_dependency_graph),
    RunnableLambda(summarize_repo)
)

fallback_chain = RunnableLambda(lambda _: {"error": "Unsupported input type"})

input_detector = RunnableLambda(detect_input_type)

# ✅ Step 2: RunnableBranch for langchain_core 1.0.3
branch_flow = RunnableBranch(
    lambda x: x["input_type"] == "file", file_flow,
    lambda x: x["input_type"] == "repo", repo_flow,
    fallback_chain  # this acts as the default branch
)

main_flow = RunnableSequence(
    input_detector,
    branch_flow
)






# ─────────────────────────────────────────────
# 5️⃣ Streamlit UI
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
    repo_url = st.text_input("Enter GitHub Repo URL:")
    if repo_url and st.button("Analyze Repository"):
        result = main_flow.invoke({"input_path": repo_url})
        st.markdown(result["final_summary"])
        st.markdown("### Dependency Graph")
        st.components.v1.html(open(result["graph_path"]).read(), height=450)
