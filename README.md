<h1 align="center">🎙️ Voice Search Enhancement Using LLMs</h1>
<p align="center">
Exploring the untapped potential of voice-based search in the e-commerce ecosystem.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-%233776AB.svg?style=for-the-badge&logo=Python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-%23FF0000.svg?style=for-the-badge&logo=Streamlit&logoColor=red&color=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/HuggingFace-%23FFB000.svg?style=for-the-badge&logo=HuggingFace&logoColor=black" alt="Hugging Face">
  <img src="https://img.shields.io/badge/Groq-%23000000.svg?style=for-the-badge&logoColor=white" alt="Groq API">
  <img src="https://img.shields.io/badge/FAISS-%23150458.svg?style=for-the-badge&logoColor=white" alt="FAISS">
</p>

---

# Voice Search Enhancement Using LLMs

This repository, `voice-search-enhancement-using-llms`, aims to leverage new Large Language Models (LLMs) and concepts to enhance voice search, especially on e-commerce platforms. The primary focus is on the ability to extract relevant results based on long, conversational audio queries.

## 🧭 Context

Search has been, from the very advent of internet marketplaces, especially in the e-commerce sector, a topic of great research and development, showcasing consistent scopes of improvement as technologies continue to rise. However, as the internet at this point in time has come to spread to the remotest of places, the diversity of the audience on these platforms has significantly increased as well. With a major portion of this audience being taken up by a group of nascent users, it has become essential to introduce more flexibility to the existing search systems to make product browsing easy and convenient. The recent rise in the NLP (Natural Language Processing) abilities of LLMs (Large Language Models), particularly in terms of conversational and multimodal capabilities, shows great promise to introduce such flexibility to the users in the form of voice-based conversational search query refinement. This project aims at achieving the same via a simple Streamlit application. 

---

### 💡 Additional Context and Motivation

While conceptualizing this project, I was particularly intrigued by the **idle microphone icons** seen across major e-commerce platforms like Amazon, Walmart, and Flipkart — elements that have remained **underutilized** despite their potential.  
I observed that such voice search features are often neglected or even removed because users rarely engage with them. However, there lies a **symbiotic opportunity** between this feature and **new or undereducated users**, especially in emerging markets like India.  

Not all users are fluent with keyword-based searches or product naming conventions. For them, **conversational queries** (“I want to gift my dad a red watch under Rs.2000”) feel natural and intuitive — much like speaking to a salesperson in an offline store.  
Current voice-search implementations only transcribe such inputs verbatim, limiting their usefulness. The lack of semantic understanding stems largely from how **relational databases** handle product information — rigid and dependent on exact term matches.

To bridge this gap, I leveraged advancements in **speech recognition (Whisper)**, **natural language reasoning (LLMs)**, and **vector databases (FAISS)** to build a pipeline that transforms long, loosely structured speech into concise semantic queries for accurate product retrieval.  
The system demonstrated **~80% relevance accuracy**, showing how **LLMs and vectorization** can revive an overlooked UX feature into something genuinely inclusive and intelligent.

---

## 📦 Data Collection and Pre-processing

Before beginning to build the project, the first and foremost action item was to find an appropriate e-commerce dataset with a diverse product catalogue to test the ability of the model to precisely fetch relevant results. Upon much research, I found the Flipkart (a prominent e-commerce platform in India) product catalogue on Data World's website (https://data.world/promptcloud/product-details-on-flipkart-com). The dataset was in a CSV format, and had about 20K data points (products) with 15 columns (attributes) consisting of multiple information on each product such as its purchase URL, category tree, price, etc.  

Once the dataset was finalized, it was now time to clean it for further processing. By leveraging simple data cleaning techniques, I was able to create a cleaner version of the CSV. 
Next, since I was trying to leverage an index-based search instead of a simple CSV search, it was time to convert the data into a search index, by embedding the data associated with each product in the dataset. For this purpose, I decided to go ahead with the FAISS (Facebook AI Similarity Search) library, since the size of the data was small and creating an entire vector database (using libraries such as Pinecone), would have been meaningless. For creating the index, I first converted my CSV into a JSON file for the ease of conversion. Next, for each item (product) in the newly-created JSON file, I kept only the relevant information (name, description, URL, etc.) and encoded the rest with the help of the 'all-MiniLM-L6-v2' model offered by the Sentence Transformers library in Python, and created embeddings for each product. 

Once the embeddings were ready, I fed them to a flat FAISS index and saved the index file locally. The rest of the functions, particularly product search, were carried out on this index file.
For your reference, these data pre-processing and conversion scripts can be found in the Misc folder of the project directory.  

---

## 🧩 Backend Script

### 🎙️ Taking the User's Audio Input

For this project, the spinal element was to be able to take the user's input in an audio format and transcribe it properly before searching for pivotal elements. For this, I leveraged the following Python libraries: torch, torchaudio, pyaudio, and wave for taking and managing audio input, and of course, transformers for converting the audio into a text transcription. The audio libraries enabled me to record the user's audio input with the host's audio input devices, automatically stop the recording after detecting a continuous silence for a buffer period (set to 4 seconds), save the audio file (as a temp file) in the local directory till it was transcribed, and finally delete it once the transcription was done. 
The audio file was transcribed to text with the help of HuggingFace and OpenAI's Whisper ('whisper-small') model leveraged using the transformers library in Python. The model conveniently converted the audio content into text, which was pushed further into the pipeline. 
These actions were taken care of within the speech_to_text.py script, and the model was initiated in the app.py script.

---

### 🧠 Converting the Transcription into Crisp Queries

Once the audio transcription was done, it was time to convert it to crisp and concise queries that FAISS's search functionality could understand and interpret. For the same, I leveraged Meta's Llama3-70b-8192 model on Groq API by using prompt engineering techniques for the system prompt (QUARRY PROMPT) and passing the transcription as the user prompt to the LLM instance. The detailed system prompt ensured the extraction of key elements from the user's audio transcription and its conversion to a proper format for FAISS. The temperature and the top-p values of the LLM instance were set to low values to ensure less entropy in the generated query. 

---

### 🔍 Getting and Cleaning the Outputs

Once the crisp and concise query for the product catalogue index (locally saved index file) was ready, the query was passed to the index search functionality of the FAISS library, which then searched the index and returned top-k (k=5) relevant and matching results, in a JSON format. 
This output JSON format, however, would be difficult for the user to read directly and grasp important information. For this, I once again leveraged another LLM instance with the same properties as the first one, and cleaned the generated JSON into a clean, readable format. The system prompt for this was the RE-SEARCHER PROMPT. 

The steps covered within the previous two sections were collectively carried out with the collaborative functioning of the prompts.py, get_search_dict.py, and query_processing.py scripts. 

---

## 🧭 High-level Workflow

![Workflow Image](./Model_Workflow.png)

The high-level workflow is illustrated above and can be summarized as follows:
1. **User Input**: The user's voice input is taken and checked for silence.
2. **Transcription**: If silence is detected for more than 4 seconds, the input is sent to OpenAI's Whisper model for speech-to-text conversion.
3. **Query Conversion**: The transcribed text is processed by the QUARRY PROMPT to convert it into a concise search query.
4. **FAISS Search**: The concise query is used to search the FAISS index.
5. **Result Cleaning**: The search results are cleaned and prettified using the RE-SEARCHER PROMPT.
6. **User Output**: The cleaned results are presented to the user.

---

## 🖥️ App Deployment

With the completion of the backend pipeline, I moved to deploy the script with a user-friendly and minimal interface using the Streamlit library in Python. Adding some simple visual elements, and arranging the product information in a readable way, I was able to make a simple UI for the model and deploy it with the help of Streamlit. For the styles, I created a styles.css file, and finally brought it all together in the app.py file to run and test.

---

## 🚀 Usage

If you wish to see the app in action, you can run it locally using the following commands:

### For Windows
```bash
streamlit run app.py
```

### For MacOS/Linux
```bash
streamlit run app.py
```

Make sure to update the keys in the .env files with the real keys, depending on whether you're using OpenAI or Groq.

---

## 🌱 Future Scopes

The current project just covers a particular use case of enhancing voice-based search queries, which is translating conversational audio queries (in English). Future implementations could include the functionality of multilingual inputs and the ability to translate them to appropriate search queries, being able to converse back in audio format, etc. As LLM techniques continue to improve, we can expect search technologies to experience an upgrade as well.

---

### 🔮 Additional Vision

This project also opens doors to several research and deployment directions — from **linguistically diverse voice search systems** to **fully conversational commerce** experiences. A multilingual extension can empower users from different regions to search naturally in their native tongues.  

Beyond the technical pipeline, this project represents an effort to **redefine accessibility in digital retail** — transforming a once “decorative” UI icon into a meaningful, human-centric interaction point.

---

## 💬 Thank You

<p align="center">
  🙏 Thank you for exploring <b>Voice Search Enhancement Using LLMs</b>!  
</p>
<p align="center">
  This project reflects an original attempt to revive an <i>underutilized UI element</i> — the voice search — through the combined power of <b>LLMs, audio processing, and vector databases</b>.  
</p>
<p align="center">
  <i>“Bringing warmth, conversation, and inclusivity back to online search.”</i> ✨  
</p>
