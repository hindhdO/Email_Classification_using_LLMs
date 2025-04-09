import streamlit as st
from main import run, run_test, compute_metrics
import json

st.set_page_config(page_title="Email Classifier", layout="centered")
st.title("📧 Smart Email Classifier")

st.markdown("Classify and evaluate emails using reasoning from large language models.")
st.markdown("---")

email = st.chat_input("💬 Enter the email you want to classify:")

if email:
    with st.spinner("🔄 Running classification and evaluation..."):
        result = run(email)

        classification = result.get("refinement_output") or json.loads(result.get("classification_output"))
        evaluation = json.loads(result.get("evaluation_output"))

    st.markdown("## ✅ Final Verdict")

    if classification:
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Spam?", value="🚫 Yes" if classification.get("spam") else "📩 No")
        with col2:
            st.metric(label="Category", value=classification.get("category"))

        with st.expander("🔍 Classifier Reasoning", expanded=True):
            st.json(classification)
    else:
        st.warning("⚠️ No classification result available.")

    st.markdown("---")

    if evaluation:
        with st.expander("🧠 Evaluator Analysis", expanded=True):
            st.json(evaluation)

        if evaluation.get("verdict") in ["Incorrect", "Partially correct"]:
            st.warning("🔁 The evaluator disagreed — refinement was applied.")
        else:
            st.success("✅ Evaluator agrees with the classification. No refinement needed.")
    else:
        st.warning("⚠️ No evaluator output available.")

num_tests = st.number_input(
    "How many emails to generate?",
    min_value=1,
    value=10,  
    step=1
)

if st.button("Launch test cases"):
    with st.spinner("🔄 Running classification and evaluation..."):
        test_result = run_test(num_tests)

    test_output_raw = test_result.get("test_output")
    parsed_test_output = json.loads(test_output_raw)
        
    test_cases = parsed_test_output.get("tests", [])
    score = parsed_test_output.get("score")
    suggestion = parsed_test_output.get("suggestion")

    metrics = compute_metrics(test_cases)

    st.subheader("📊 Metrics")

    ignore_keys = ["accuracy", "macro avg", "weighted avg"]
    for category, values in metrics.items():
        if category in ["Spam", "Urgent", "Not Urgent", "To Read"]:
            with st.expander(f"📧 {category}"):
                col1, col2, col3 = st.columns(3)
                col1.metric("🎯 Precision", f"{values['precision']*100:.1f}%")
                col2.metric("🎯 Recall", f"{values['recall']*100:.1f}%")
                col3.metric("🧮 F1-score", f"{values['f1-score']*100:.1f}%")

    if score is not None:
        st.success(f"✅ accuracy: {score}/{num_tests}")

    if suggestion:
        st.info(f"💡 Suggestion: {suggestion}")

    if test_cases:
        st.subheader("📄 Test Cases")
        for case in test_cases:
            with st.expander(f"Test: {case.get('expected')} → {case.get('predicted')}"):
                st.write(case)
