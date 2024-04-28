import openai
import time

import os
import pandas as pd
import pickle
import streamlit as st
from dotenv import load_dotenv
import json
from sklearn.impute import SimpleImputer

# Load environment variables
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=openai.api_key)
model = "gpt-4-turbo"

class AssistantManager:
   

    def __init__(self, model: str = model):
        self.assistant_id = self.load_assistant_id()
        self.model = model
        self.client = client
        self.thread = None
        self.run = None
        self.summary = None
        self.uploaded_file = None 
        # Retrieve existing assistant and thread if IDs are stored in session state
        if st.session_state.assistant_id:
            self.assistant = self.client.beta.assistants.retrieve(
                assistant_id=st.session_state.assistant_id
            )
        if st.session_state.thread_id:
            self.thread = self.client.beta.threads.retrieve(
                thread_id=st.session_state.thread_id
            )

    def load_assistant_id(self):
        try:
            with open('assistant.json', 'r') as file:
                data = json.load(file)
                return data['assistant_id']
        except (FileNotFoundError, KeyError):
            return None

    def create_assistant(self):
        if self.assistant_id is not None:
            return self.assistant_id

        # tools = [{
        #     "type": "function",
        #     "function": {
        #         "name": "analyze_file",
        #         "description": "analyze a file."
        #     }
        # }]
        assistant = openai.beta.assistants.create(
            name="Fraud Detection Assistant",
            description="Process the uploaded CSV to detect fraud and answer queries about the data.",
            instructions="Process the uploaded CSV to detect fraud and answer queries about the data.",
            # tools=tools,
            model=model  # Make sure this model name is available to you
        )
        self.assistant_id = assistant.id
        with open('assistant.json', 'w') as file:
            json.dump({'assistant_id': self.assistant_id}, file)
        return self.assistant_id

    def create_thread(self):
        if not self.thread:
            thread_obj = self.client.beta.threads.create()
            self.thread = thread_obj  # This sets the local thread instance
            st.session_state.thread_id = thread_obj.id  # Save the thread ID in session state
            print(f"Thread ID:::: {self.thread.id}")

    def add_message_to_thread(self, role, content):
        if self.thread:
            self.client.beta.threads.messages.create(
                thread_id=self.thread.id,
                role=role,
                content=content
            )

    def run_assistant(self, instructions):
        if self.assistant_id and self.thread:
            self.run = self.client.beta.threads.runs.create(
                thread_id=self.thread.id,
                assistant_id=self.assistant_id,
                instructions=instructions
            )

    def process_message(self):
        if self.thread:
            messages = self.client.beta.threads.messages.list(thread_id=self.thread.id)
            summary = []
            last_message = messages.data[0]
            role = last_message.role
            response = last_message.content[0].text.value
            summary.append(response)
            self.summary = "\n".join(summary)
            print(f"SUMMARY-----> {role.capitalize()}: ==> {response}")

    def wait_for_completion(self):
        if self.thread and self.run:
            while True:
                time.sleep(5)
                run_status = self.client.beta.threads.runs.retrieve(
                    thread_id=self.thread.id,
                    run_id=self.run.id
                )
                print(f"RUN STATUS::{run_status.model_dump_json(indent=4)}")
                if run_status.status == "completed":
                    self.process_message()
                    break
                elif run_status.status == "requires_action":
                    print("FUNCTION CALLING NOW....")
                    self.call_required_functions(required_actions=run_status.required_action.submit_tool_outputs.model_dump())

    def call_required_functions(self, required_actions):
        if not self.run:
            return
        tool_outputs = []
        for action in required_actions["tool_calls"]:
            func_name = action["function"]["name"]
            arguments = json.loads(action["function"]["arguments"])
            if func_name == "load_and_predict":
        # Ensure self.uploaded_file is set before this is called
                 if self.uploaded_file is not None:
                     load_file = load_and_predict(self.uploaded_file)
                     tool_outputs.append({
                        "tool_call_id": action["id"],
                        "output": json.dumps(load_file.tolist()),  # Assuming the output is JSON serializable
                    })
                 else:
                        print("No uploaded file to process.")
                
            else:
                raise ValueError(f"Unknown Function: {func_name}")
        print("Submitting outputs back to the Assistant...")
        self.client.beta.threads.runs.submit_tool_outputs(
            thread_id=self.thread.id,
            run_id=self.run.id,
            tool_outputs=tool_outputs,
        )

    def get_summary(self):
        return self.summary

    def run_steps(self):
        if not self.thread:
            print("No thread available")
            return []
    
        run_steps = self.client.beta.threads.runs.steps.list(
            thread_id=self.thread.id,
            run_id=self.run.id
        )
        print(f"Run-Steps::: {run_steps}")
        return run_steps.data
    
    def analyze_file(self):
        if self.uploaded_file is not None:
            insights = load_and_analyse(self.uploaded_file)
            self.add_message_to_thread('assistant', f"Analyzing file: {insights}")
            self.run_assistant("Provide insights based on the uploaded file")
            self.wait_for_completion()
            return self.get_summary()
        else:
            return "No file uploaded to analyze."

    



def load_and_predict(uploaded_file):
    # Ensure the file pointer is at the beginning
    uploaded_file.seek(0)

    try:
        # Read the uploaded CSV file; assuming delimiter is comma and there is a header
        df = pd.read_csv(uploaded_file)
    except pd.errors.EmptyDataError:
        return "The uploaded CSV file is empty or improperly formatted."

    # Drop the columns that are not used by the model
    columns_to_drop = ['policy_number', 'policy_bind_date', 'insured_zip', 'incident_location', 'incident_date']
    df.drop(columns=columns_to_drop, errors='ignore', inplace=True)

    # Convert all columns to numeric, handling errors by coercing to NaN
    df = df.apply(pd.to_numeric, errors='coerce')

    # Impute NaN values using a constant strategy
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    imputed_data = imputer.fit_transform(df)
    
    # Create a DataFrame with the imputed data
    df = pd.DataFrame(imputed_data, columns=df.columns)

    # Load model and predict
    model = pickle.load(open("insurance_random_forest_model_1.pkl", "rb"))
    predictions = model.predict(df)
    return predictions


def upload_file(uploaded_file):
    # Use the uploaded file object directly instead of a file path
    uploaded_file.seek(0)  # Move to the beginning of the file in case it was already read
    file_response = client.files.create(file=uploaded_file, purpose="assistants")
    return file_response.id

def load_and_analyse(uploaded_file):
    # Ensure the file pointer is at the beginning
    uploaded_file.seek(0)

    try:
        df = pd.read_csv(uploaded_file)
    except pd.errors.EmptyDataError:
        return "The uploaded CSV file is empty or improperly formatted."

    # Assuming you want to create a textual summary of the data
    textual_summary = create_text_summary(df)
    return textual_summary

def create_text_summary(df):
    # Create a summary text from the dataframe
    summary_lines = []
    for index, row in df.iterrows():
        line = f"Record {index+1}: " + ", ".join([f"{col}={val}" for col, val in row.items() if pd.notna(val)])
        summary_lines.append(line)
    summary_text = "\n".join(summary_lines)
    return summary_text

def main():
    st.title("Fraud Prediction System")
    # Initialize the assistant and thread if not already set
    if 'assistant_id' not in st.session_state or 'thread_id' not in st.session_state:
        st.session_state.assistant_id = None
        st.session_state.thread_id = None
    manager = AssistantManager()
    if not st.session_state.assistant_id:
        assistant_id = manager.create_assistant()
        st.session_state.assistant_id = assistant_id
    if not st.session_state.thread_id:
        manager.create_thread()
        st.session_state.thread_id = manager.thread.id

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display existing conversation messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
     # User input through chat interface
    user_input = st.chat_input("How can I assist you with your today?")
    if user_input:
        # Append user input to the session state messages
        st.session_state.messages.append({"role": "user", "content": user_input})
        # Immediately display user input
        with st.chat_message("user"):
            st.markdown(user_input)

        # Add message to thread and handle the input
        manager.add_message_to_thread(role="user", content=user_input)
        # manager.add_message_to_thread("user", f"Predictions: {prediction_output}")
        manager.run_assistant(instructions="Handle the user's input")
        manager.wait_for_completion()

        # Get and display the summary response from the assistant
        recommendations = manager.get_summary()
        st.session_state.messages.append({"role": "assistant", "content": recommendations})
        with st.chat_message("assistant"):
            st.markdown(recommendations)

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
         
        
        manager.uploaded_file = uploaded_file 
        insights = manager.analyze_file()
        st.write(insights)
        # Process the predictions
        predictions = load_and_predict(uploaded_file)

        # Convert predictions to a DataFrame and then to CSV for download
        predictions_df = pd.DataFrame(predictions, columns=['Predictions'])
        predictions_csv = predictions_df.to_csv(index=False).encode('utf-8')
        
        # Use Streamlit to download the result as a CSV file
        st.download_button(
            label="Download Predictions as CSV",
            data=predictions_csv,
            file_name='predictions.csv',
            mime='text/csv',
        )

       
        
        # Use the original uploaded file to upload to OpenAI's servers
        uploaded_file.seek(0)  # Reset file pointer to the beginning
        file_id = upload_file(uploaded_file)  # Use modified upload function

        # Assuming you want to add the predictions as a message to the thread
        prediction_output = ", ".join(map(str, predictions))
        #manager.add_message_to_thread(thread_id, "user", f"Predictions: {prediction_output}")
        
       
        
    #     # run_id = manager.run_assistant(thread_id, assistant_id)
    #     # st.write(f"Run ID: {run_id}")



if __name__ == "__main__":
    main()