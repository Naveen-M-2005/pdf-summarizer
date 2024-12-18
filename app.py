import os
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from flask_session import Session
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import sqlite3
import logging
from datetime import datetime

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['UPLOAD_FOLDER'] = './uploads'
Session(app)

# API key setup
api_key = "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"
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# LLM setup
llm = ChatGroq(groq_api_key=api_key, model_name="Llama-3.1-8b-Instant")
vectorstore = None
history_aware_retriever = None
conversational_rag_chain = None

def init_db():
    if not os.path.exists('users.db'):
        with sqlite3.connect('users.db') as conn:
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                              id INTEGER PRIMARY KEY AUTOINCREMENT,
                              email TEXT UNIQUE NOT NULL,
                              password TEXT NOT NULL)''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS history (
                              id INTEGER PRIMARY KEY AUTOINCREMENT,
                              user_id INTEGER,
                              question TEXT,
                              answer TEXT,
                              language TEXT,
                              timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                              FOREIGN KEY (user_id) REFERENCES users (id))''')
            conn.commit()

# Initialize database
init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        hashed_password = generate_password_hash(password)
        with sqlite3.connect('users.db') as conn:
            cursor = conn.cursor()
            try:
                cursor.execute('''INSERT INTO users (email, password) VALUES (?, ?)''', (email, hashed_password))
                conn.commit()
                flash('Signup successful! Please log in.', 'success')
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                flash('Email already exists!', 'error')
                return render_template("signup.html")
    return render_template("signup.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        with sqlite3.connect('users.db') as conn:
            cursor = conn.cursor()
            cursor.execute('''SELECT * FROM users WHERE email = ?''', (email,))
            user = cursor.fetchone()
            if user and check_password_hash(user[2], password):
                session['user_id'] = user[0]
                session['logged_in'] = True
                return redirect(url_for('upload_pdf'))
            else:
                flash('Invalid email or password!', 'error')
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop('user_id', None)
    session.pop('logged_in', None)
    return redirect(url_for('index'))

@app.route('/upload_pdf', methods=['GET', 'POST'])
def upload_pdf():
    global vectorstore, history_aware_retriever, conversational_rag_chain

    if 'logged_in' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)

        files = request.files.getlist('file')
        documents = []

        for file in files:
            if file.filename == '':
                flash('No selected file', 'error')
                return redirect(request.url)
            
            if file and file.filename.endswith('.pdf'):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                loader = PyPDFLoader(filepath)
                docs = loader.load()
                documents.extend(docs)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in session:
                session[session_id] = ChatMessageHistory()
            return session[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        return redirect(url_for('ask_question'))

    return render_template('upload_pdf.html')

@app.route('/ask_question', methods=['GET', 'POST'])
def ask_question():
    global conversational_rag_chain
    if request.method == 'POST':
        question = request.form['question']
        language = request.form['language']

        if not conversational_rag_chain:
            flash('Please upload a PDF first.', 'error')
            return redirect(url_for('upload_pdf'))

        session_id = 'default_session'

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in session:
                session[session_id] = ChatMessageHistory()
            return session[session_id]

        session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input": question, "language": language},
            config={
                "configurable": {"session_id": session_id}
            }
        )

        # Save to history
        user_id = session['user_id']
        with sqlite3.connect('users.db') as conn:
            cursor = conn.cursor()
            cursor.execute('''INSERT INTO history (user_id, question, answer, language, timestamp)
                              VALUES (?, ?, ?, ?, ?)''', 
                              (user_id, question, response['answer'], language, datetime.now()))
            conn.commit()

        return render_template('result.html', answer=response['answer'])
    
    return render_template('ask_question.html', languages=["English", "Spanish", "French", "German"])

@app.route('/history', methods=['GET', 'POST'])
def history():
    if 'logged_in' not in session:
        flash('You need to be logged in to view your history.', 'error')
        return redirect(url_for('login'))

    user_id = session.get('user_id')

    if request.method == 'POST':
        history_id = request.form['history_id']
        try:
            with sqlite3.connect('users.db') as conn:
                cursor = conn.cursor()
                cursor.execute('''DELETE FROM history WHERE id = ? AND user_id = ?''', (history_id, user_id))
                conn.commit()
                flash('History entry deleted successfully.', 'success')
        except sqlite3.Error as e:
            logging.error(f"Database error: {e}")
            flash('An error occurred while deleting the history entry. Please try again later.', 'error')

    try:
        with sqlite3.connect('users.db') as conn:
            cursor = conn.cursor()
            cursor.execute('''SELECT id, question, answer, language, timestamp 
                              FROM history 
                              WHERE user_id = ? 
                              ORDER BY timestamp DESC''', (user_id,))
            history_data = cursor.fetchall()

        if not history_data:
            flash('No history found for your account.', 'info')

        return render_template('history.html', history=history_data)

    except sqlite3.OperationalError as e:
        logging.error(f"Database operational error: {e}")
        flash('Database error: Please check the database setup.', 'error')
        return redirect(url_for('index'))
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        flash('An error occurred while retrieving your history. Please try again later.', 'error')
        return redirect(url_for('index'))
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        flash('An unexpected error occurred. Please try again later.', 'error')
        return redirect(url_for('index'))
    
@app.route('/send-message', methods=['POST'])
def send_message():
    name = request.form.get('name')
    email = request.form.get('email')
    message = request.form.get('message')
    print(f"Received message from {name} ({email}): {message}")
    return jsonify({'message': 'Message sent successfully!', 'status': 'success'})


if __name__ == '__main__':
    init_db()  # Initialize database and tables
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    logging.basicConfig(level=logging.DEBUG)
    app.run(debug=False, use_reloader=False)
