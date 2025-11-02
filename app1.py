import streamlit as st
import cohere
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("COHERE_API_KEY")

st.set_page_config(
    page_title="GenAI Code Context & Improvement Assistant",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 GenAI Code Context & Improvement Assistant")
st.write("Upload a code file to get **context summary** and **suggested improvements** using Cohere's Command-A model.")

# Check API key
if not api_key:
    st.error("⚠️ Cohere API key not found! Please set COHERE_API_KEY in your .env file.")
    st.stop()

# Initialize Cohere client (v5+)
co = cohere.ClientV2(api_key)

uploaded_file = st.file_uploader(
    "Upload your code file (.py, .js, .java, .cpp, .html, .ts, .txt, etc.)",
    type=["py", "js", "java", "cpp", "html", "ts", "txt", "c"],
)

if uploaded_file is not None:
    code_content = uploaded_file.read().decode("utf-8")

    st.subheader("📂 Uploaded Code Preview")
    st.code(code_content, language="python")

    if st.button("🔍 Analyze Code"):
        with st.spinner("Analyzing your code with Cohere Command-A..."):
            try:
                response = co.chat(
                    model="command-a-03-2025",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert code review and optimization assistant. "
                                "Analyze user-provided code and explain: "
                                "1️⃣ What it does (context summary)\n"
                                "2️⃣ How it can be improved (readability, performance, structure)."
                            ),
                        },
                        {"role": "user", "content": code_content},
                    ],
                )

                result_text = response.message.content[0].text.strip()

                # Optional: split into sections
                if "Suggestions:" in result_text:
                    context, suggestions = result_text.split("Suggestions:", 1)
                else:
                    context, suggestions = result_text, ""

                st.success("✅ Analysis Complete!")

                with st.expander("📘 Code Context (Summary)", expanded=True):
                    st.write(context.strip())

                with st.expander("🧩 Improvement Suggestions", expanded=True):
                    st.write(suggestions.strip() if suggestions else "No clear suggestions found.")

            except Exception as e:
                st.error(f"❌ Error during Cohere API call: {str(e)}")
else:
    st.info("Please upload a code file to begin analysis.")
