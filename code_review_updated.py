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
    st.subheader("📁 Analyze a GitHub Repository")

    # Helper: list supported files recursively
    def get_repo_files(repo_path):
        supported_exts = (
            ".py", ".js", ".cpp", ".java", ".ipynb",
            ".csv", ".sh", ".json", ".html", ".md", ".txt"
        )
        repo_files = []
        for root, _, files in os.walk(repo_path):
            for f in files:
                if f.endswith(supported_exts):
                    repo_files.append(os.path.join(root, f))
        return repo_files

    repo_url = st.text_input("Enter GitHub Repo URL:")

    # Step 1: Load and analyze repo structure
    if st.button("🔍 Load Repository"):
        with st.spinner("⏳ Cloning repository and analyzing structure..."):
            base_flow = RunnableSequence(
                RunnableLambda(load_repo),
                RunnableLambda(analyze_repo_structure)
            )
            base_result = base_flow.invoke({"input_path": repo_url})
            st.session_state["repo_data"] = base_result
            st.success("✅ Repository loaded successfully!")

    # Step 2: Show file choices after repo is loaded
    if "repo_data" in st.session_state:
        repo_data = st.session_state["repo_data"]
        repo_structure = repo_data["repo_structure"]
        all_files = sum(repo_structure.values(), [])

        st.write("### Repository contains:")
        st.json(repo_structure)

        choice = st.radio("Choose analysis mode:", ("Analyze all files", "Select specific file"))

        if choice == "Select specific file":
            selected = st.selectbox("Select a file to analyze:", all_files)
            selected_files = [selected]
        else:
            selected_files = all_files

        # Step 3: Analyze button
        if st.button("🚀 Run Analysis"):
            with st.spinner("🧠 Analyzing code... this may take a few moments"):
                inputs = {
                    **repo_data,
                    "selected_files": selected_files,
                    "analysis_mode": "specific" if choice == "Select specific file" else "all",
                }

                # Ensure selected_files exist
                if not inputs.get("selected_files"):
                    print("⚠️ No specific files selected — analyzing all supported files in repo.")
                    repo_files = get_repo_files(inputs["repo_path"])
                    inputs["selected_files"] = repo_files

                # Build and run analysis flow
                analysis_flow = RunnableSequence(
                    RunnableLambda(analyze_files),
                    RunnableLambda(build_dependency_graph),
                    RunnableLambda(summarize_repo)
                )

                result = analysis_flow.invoke(inputs)
                st.session_state["analysis_result"] = result
                st.success("✅ Analysis complete!")

    # Step 4: Display results
    if "analysis_result" in st.session_state:
        result = st.session_state["analysis_result"]
        st.markdown("## 🧾 Final Repository Summary")
        st.markdown(result.get("final_summary", "No summary generated."))
        st.markdown("### 🔗 Dependency Graph")
        st.components.v1.html(open(result["graph_path"]).read(), height=450)
