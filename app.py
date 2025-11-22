# import os
# import streamlit as st
# from groq import Groq

# # Page Configuration
# st.set_page_config(page_title="AI-Powered Coding Assistant", layout="wide")

# # Sidebar Initialization
# with st.sidebar:
#     st.title("🔑 API Configuration")
#     st.write("Enter your Groq API Key to start using the assistant.")
#     groq_api_key = st.text_input("Groq API Key", value="", type="password")
# groq_client = Groq(api_key=groq_api_key)

# # Function to generate a response using the GROQ chat completion API
# def generate_response(prompt):
#     response = groq_client.chat.completions.create(
#         messages=[
#             {
#                 "role": "user",
#                 "content": prompt,
#             }
#         ],
#         model="llama3-8b-8192",
#     )
#     return response.choices[0].message.content

# # Function to generate code completion
# def generate_code_completion(code_input):
#     prompt = f"You are a coding assistant.Help students understand how to code. Suggest the next line for this code:\n{code_input}"
#     return generate_response(prompt)

# # Function to detect errors in code
# def detect_errors_in_code(code):
#     prompt = f"Act like a complier and Detect errors in this code:\n{code}. Suggest the required changes."
#     return generate_response(prompt)

# # Function to explain code
# def explain_code(code):
#     prompt = f"You are an Coding assistant. Explain this code line by line:\n{code}"
#     return generate_response(prompt)

# # Suggest Similar questions
# def suggest_question(code):
#     prompt = f"You are an Coding expert. Based on the code provided :\n{code}. Suggest 2-3 pratise questions similar to the code for logic building and understanding the concept "
#     return generate_response(prompt)

# # Function to generate code from a problem description
# def generate_code_from_description(description):
#     prompt = f"You are a coding expert. Write code to {description}"
#     return generate_response(prompt)

# # Function to answer questions about coding
# def ask_question(question):
#     prompt = f"You are an expert in coding. Explain {question} in detail with example"
#     return generate_response(prompt)

# def main():
#     st.title("🤖 AI-Powered Coding Assistant")
#     st.markdown("Your personal AI helper for coding suggestions, error detection, explanations, and more!")
    
#     # Code Completion Section
#     with st.expander("💡 Code Completion", expanded=True):
#         st.subheader("Suggest the Next Line for Your Code")
#         code_input = st.text_area("Write your code here:")
#         if st.button("Suggest Next Line"):
#             if code_input:
#                 with st.spinner("Generating suggestion..."):
#                     suggestion = generate_code_completion(code_input)
#                 st.write(f"Suggestion: {suggestion}")
#             else:
#                 st.warning("Please enter some code to get a suggestion.")

#     # Error Detection Section
#     with st.expander("🐞 Error Detection", expanded=False):
#         st.subheader("Find Errors in Your Code")
#         if st.button("Check for Errors"):
#             if code_input:
#                 with st.spinner("Detecting errors..."):
#                     error_report = detect_errors_in_code(code_input)
#                 st.write(f"Error Report:\n{error_report}")
#             else:
#                 st.warning("Please enter some code to check for errors.")

#     # Code Explanation Section
#     with st.expander("📘 Code Explanation", expanded=False):
#         st.subheader("Get Explanation for Your Code")
#         if st.button("Explain Code"):
#             if code_input:
#                 with st.spinner("Generating explanation..."):
#                     explanation = explain_code(code_input)
#                 st.info(f"Explanation:\n{explanation}")
#             else:
#                 st.warning("Please enter some code to explain.")

#     # Practice Questions Section
#     with st.expander("📝 Practise Questions", expanded=False):
#         st.subheader("Get Practice Questions for Your Code")
#         if st.button("Questions"):
#             if code_input:
#                 with st.spinner("Generating questions..."):
#                     questions = suggest_question(code_input)
#                 st.info(f"Practise questions:\n{questions}")
#             else:
#                 st.warning("Please enter some code to suggest similar questions.")

#     # Code Generation from Description Section
#     with st.expander("💻 Code Generation from Description", expanded=False):
#         st.subheader("Generate Code from a Description")
#         problem_description = st.text_input("Describe what you need the code to do:")
#         if st.button("Generate Code"):
#             if problem_description:
#                 with st.spinner("Generating code..."):
#                     code_output = generate_code_from_description(problem_description)
#                 st.code(code_output, language='python')
#             else:
#                 st.warning("Please provide a problem description.")

#     # Q&A Section
#     with st.expander("❓ Coding Q&A", expanded=False):
#         st.subheader("Ask a Coding Question")
#         question = st.text_input("Ask a coding question:")
#         if st.button("Get Answer"):
#             if question:
#                 with st.spinner("Finding answer..."):
#                     answer = ask_question(question)
#                 st.write(f"Answer:\n{answer}")
#             else:
#                 st.warning("Please ask a question.")

# # Run the app
# if __name__ == "__main__":
#     main()


import cv2
print("OpenCV version:", cv2.__version__)



 