from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType
from twelvelabs import TwelveLabs
from twelvelabs.models.embed import SegmentEmbedding
import os
import google.generativeai as genai
import streamlit as st

# Zilliz/Milvus configuration
ZILLIZ_CLOUD_URI = ""
ZILLIZ_CLOUD_API_KEY = ""
TWELVE_LABS_API_KEY = ""
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Initialize embeddings
huggingface_model_path = "sentence-transformers/all-MiniLM-L6-v2"
hf_model_args = {"device": "cpu"}  # Adjust device if GPU is available
hf_embeddings = HuggingFaceEmbeddings(
    model_name=huggingface_model_path,
    model_kwargs=hf_model_args,
    encode_kwargs={"normalize_embeddings": True}
)

# Twelve Labs client
twelvelabs_client = TwelveLabs(api_key=TWELVE_LABS_API_KEY)

# PDF and video configurations
pdf_directory_ws = "/Users/mrunmayeerane/Desktop/hackathon/women-in-ai-hackathon/src/data/pdfs/"
video_directory_ws = "/Users/mrunmayeerane/Desktop/hackathon/women-in-ai-hackathon/src/data/"
pdf_collection_name_ws = "pdf_embeddings_collection"
video_collection_name_ws = "video_embeddings_collection"

pdf_directory_cc = "/Users/mrunmayeerane/Desktop/hackathon/women-in-ai-hackathon/src/data/courses/pdfs/"
video_directory_cc = "/Users/mrunmayeerane/Desktop/hackathon/women-in-ai-hackathon/src/data/courses/courses_video/"
pdf_collection_name_cc = "Courses_pdf_embeddings_collection"
video_collection_name_cc = "Courses_video_embeddings_collection"

# Mapping video filenames to URLs
video_urls_ws = {
    "Howtonotgetoverlookduringpromo.mp4": "https://drive.google.com/file/d/1bpu4G74xaOv7mH5ze_G_j11FqDBmSWD7/view?usp=sharing",
    "Stuck_in_career.mp4": "https://drive.google.com/file/d/1LjiHw5oscH8iiChQ71auVd-2o0Uo1_8Z/view?usp=sharing",
    "Start_overthinking_start_writing.mp4": "https://drive.google.com/file/d/1ZPYSkk8-Eti7r135z1CJWucVano5Y-9y/view?usp=sharing",
    "How_to_give_Feed_Back.mp4": "https://drive.google.com/file/d/1UhGVRHDpFG7sGav1OoGqLZp4In7YGgHM/view?usp=sharing",
    "How_to_be_assertive_at_work.mp4": "https://drive.google.com/file/d/1pDl6n0lr_6aSLldbO6GFtm9ukj2lnVzg/view?usp=sharing",
    "How_to_respond to_office_gossip.mp4": "https://drive.google.com/file/d/1MB1AM7rUrIuxAQDGsejB1Vo7gjNUn4kC/view?usp=sharing",
    "No_Sacrifice_No_Success.mp4": "https://drive.google.com/file/d/1FOLjCGgRRANsA0LZ6leo6j6KshybL70S/view?usp=sharing",
}
video_urls_cc = {
    "V1.mp4": "https://drive.google.com/file/d/1xNJ6XxNJdmbzYaCA57dRtKqY6hLaw7Bi/view?usp=sharing",
    "V2.mp4": "https://drive.google.com/file/d/1-JtxoJYV1ZUq3i7oW4s4VcWeTMIRNdwn/view?usp=sharing",
}

pdfs_urls = {
    "Insights_Techniques_and_Evaluation_for_LLM-Driven_Knowledg_Graphs _NVIDIA_Technical_Blog.pdf": "https://developer.nvidia.com/blog/insights-techniques-and-evaluation-for-llm-driven-knowledge-graphs/",
    "Optimize_AI_Inference_Performance_with_NVIDIA_Full_Stack_Solutions_ NVIDIA_Technical_Blog.pdf":"https://developer.nvidia.com/blog/optimize-ai-inference-performance-with-nvidia-full-stack-solutions/",
    "Building_a_Multimodal_Retrieval_Augmented_Generation_Application_with_Twelve_Labs_and_Milvus.pdf": "https://www.twelvelabs.io/blog/fashion-chat-assistant",
    "Building_a_Security_Analysis Application_with_Twelve_Labs.pdf": "https://www.twelvelabs.io/blog/security-analysis",
}


# Zilliz/Milvus connection
connections.connect(
    alias="default",
    uri=ZILLIZ_CLOUD_URI,
    token=ZILLIZ_CLOUD_API_KEY
)
# code 1
# Utility: Text extraction from PDF
def extract_text_from_pdf_ws(pdf_path):
    reader = PdfReader(pdf_path)
    return "".join(page.extract_text() for page in reader.pages)

# Create PDF embeddings
def create_pdf_embeddings_ws():
    if utility.has_collection(pdf_collection_name_ws):
        utility.drop_collection(pdf_collection_name_ws)

    schema = CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
            FieldSchema(name="chunk", dtype=DataType.VARCHAR, max_length=600),
            FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=256),
        ],
        description="PDF document embeddings"
    )
    collection = Collection(name=pdf_collection_name_ws, schema=schema)

    for pdf_file in os.listdir(pdf_directory_ws):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory_ws, pdf_file)
            text = extract_text_from_pdf_ws(pdf_path)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.split_text(text)
            embeddings = hf_embeddings.embed_documents(chunks)
            data = [embeddings, chunks, [pdf_file] * len(chunks)]
            collection.insert(data)
    collection.create_index(
        field_name="embedding",
        index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}}
    )
    collection.load()
    
# Create video embeddings
def create_video_embeddings_ws():
    if utility.has_collection(video_collection_name_ws):
        utility.drop_collection(video_collection_name_ws)

    schema = CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
            FieldSchema(name="start_offset_sec", dtype=DataType.FLOAT),
            FieldSchema(name="end_offset_sec", dtype=DataType.FLOAT),
            FieldSchema(name="video_name", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="video_url", dtype=DataType.VARCHAR, max_length=512),
        ],
        description="Video embeddings"
    )
    collection = Collection(name=video_collection_name_ws, schema=schema)

    for video_file in os.listdir(video_directory_ws):
        if video_file.endswith(".mp4"):
            with open(os.path.join(video_directory_ws, video_file), "rb") as video:
                task = twelvelabs_client.embed.task.create(
                    model_name="Marengo-retrieval-2.7",
                    video_file=video
                )
                task.wait_for_done(timeout=1200)
                if task.status == "ready":
                    segments = task.retrieve().video_embedding.segments
                    embeddings = [seg.embeddings_float for seg in segments]
                    start_offsets = [seg.start_offset_sec for seg in segments]
                    end_offsets = [seg.end_offset_sec for seg in segments]
                    video_name = os.path.basename(video_file)
                    video_urls_list = [video_urls_ws.get(video_name, "Unknown")] * len(segments)
                    data = [embeddings, start_offsets, end_offsets, [video_name] * len(segments), video_urls_list]
                    collection.insert(data)
    collection.create_index(
        field_name="embedding",
        index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}}
    )
    collection.load()

def perform_similarity_search_ws(query, collection_name, expected_dim, embedding_model, top_k=3):
    collection = Collection(name=collection_name)
    if embedding_model == "hf":
        query_embedding = hf_embeddings.embed_query(query)
    elif embedding_model == "twelvelabs":
        query_embedding = twelvelabs_client.embed.create(
            model_name="Marengo-retrieval-2.7",
            text=query
        ).text_embedding.segments[0].embeddings_float
    else:
        raise ValueError("Invalid embedding model specified.")

    # Define output fields based on collection type
    output_fields = ["chunk", "file_name"] if collection_name == pdf_collection_name_ws else ["video_url", "video_name", "start_offset_sec", "end_offset_sec"]

    return collection.search(
        data=[query_embedding[:expected_dim]],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=output_fields
    )

# Retrieve results
def retrieve_results_ws(query):
    pdf_results = perform_similarity_search_ws(query, pdf_collection_name_ws, 384, "hf")
    video_results = perform_similarity_search_ws(query, video_collection_name_ws, 1024, "twelvelabs")

    pdf_urls = [{"chunk": match.entity.get("chunk"), "file": match.entity.get("file_name")} for result in pdf_results for match in result]
    video_urls = [
        {
            "video_url": match.entity.get("video_url"),
            "video_name": match.entity.get("video_name"),
            "start_offset_sec": match.entity.get("start_offset_sec"),
            "end_offset_sec": match.entity.get("end_offset_sec")
        }
        for result in video_results for match in result
    ]
    print(video_urls)
    return pdf_urls, video_urls

# Intent Classification: Career vs. Workplace Help
def classify_query_intent(query):
    """
    Classifies the intent of the user's query as either 'Career Transition' or 'Workplace Stress'.
    """
    # Sample classification prompt
    classification_prompt = f"""
    You are an intent classifier for routing queries.
    Classify the following query as one of two categories:
    - 'Career Transition' if it is about changing careers, learning paths, or career development.
    - 'Workplace Stress' if it is about dealing with stress, anxiety, workplace behavior, or related topics.

    Query: "{query}"
    Response: 
    """

    model = genai.GenerativeModel("gemini-2.0-flash-exp")
    response = model.generate_content(classification_prompt).text.strip().lower()

    if "career transition" in response:
        return "Career Transition"
    elif "workplace stress" in response:
        return "Workplace Stress"
    else:
        return "Career Transition"

# Code 1: Career Coaching Advisor
def handle_workplace_query(query):
    """
    Handles workplace stress queries using the logic from Code 1.
    """
    print("Routing to Workplace Stress Logic...")
    def query_llm_ws(query):
        pdf_urls, video_urls = retrieve_results_ws(query)
        system_prompt = "You are a great career advisor and your goal is to help people to cope with work stress/anxiety and grow in their careers."
        llm_prompt = f"{system_prompt}\n\nFor the query '{query}', the following relevant resources have been retrieved:\n\n"
        llm_prompt += "Top PDFs:\n" + "\n".join([f"{i+1}. {pdf['chunk']} (File: {pdf['file']})" for i, pdf in enumerate(pdf_urls)]) + "\n\n"
        llm_prompt += "Top Videos:\n" + "\n".join([
            f"{i+1}. {video['video_url']} (Video: {video['video_name']}, Start: {video['start_offset_sec']}s, End: {video['end_offset_sec']}s)"
            for i, video in enumerate(video_urls)
        ])
    
        # print("LLM Prompt:\n", llm_prompt)
    
        # LLM Integration (Example with Generative AI)
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        response = model.generate_content(llm_prompt)
    
        # Adding the retrieved video details to the response
        response_text = response.text + "\n\nRetrieved Video Details:\n"
        response_text += "\n".join([
            f"{i+1}. ![Video Thumbnail](https://img.youtube.com/vi/{video['video_url'].split('/')[-2]}/0.jpg) "
            f"[Watch {video['video_name']}]( {video['video_url']}) "
            f"(Start: {video['start_offset_sec']}s, End: {video['end_offset_sec']}s)"
            for i, video in enumerate(video_urls)
        ])
    
        return response_text
    response = query_llm_ws(query)  # This is the function from Code 1
    return response

# Code 2: Course Generation for Career Transition
# # Utility: Text extraction from PDF
def extract_text_from_pdf_cc(pdf_path):
    reader = PdfReader(pdf_path)
    return "".join(page.extract_text() for page in reader.pages)

# Create PDF embeddings
def create_pdf_embeddings_cc():
    if utility.has_collection(pdf_collection_name_cc):
        utility.drop_collection(pdf_collection_name_cc)

    schema = CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
            FieldSchema(name="chunk", dtype=DataType.VARCHAR, max_length=600),
            FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="pdf_url", dtype=DataType.VARCHAR, max_length=512),
        ],
        description="PDF document embeddings"
    )
    collection = Collection(name=pdf_collection_name_cc, schema=schema)

    for pdf_file in os.listdir(pdf_directory_cc):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory_cc, pdf_file)
            text = extract_text_from_pdf_cc(pdf_path)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.split_text(text)
            embeddings = hf_embeddings.embed_documents(chunks)
            # Get the URL for the PDF file
            pdf_url = pdfs_urls.get(pdf_file, "Unknown")
            data = [embeddings, chunks, [pdf_file] * len(chunks), [pdf_url] * len(chunks)]
            collection.insert(data)

    collection.create_index(
        field_name="embedding",
        index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}}
    )
    collection.load()
def create_video_embeddings_cc():
    if utility.has_collection(video_collection_name_cc):
        utility.drop_collection(video_collection_name_cc)

    schema = CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
            FieldSchema(name="start_offset_sec", dtype=DataType.FLOAT),
            FieldSchema(name="end_offset_sec", dtype=DataType.FLOAT),
            FieldSchema(name="video_name", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="video_url", dtype=DataType.VARCHAR, max_length=512),
        ],
        description="Video embeddings"
    )
    collection = Collection(name=video_collection_name_cc, schema=schema)

    for video_file in os.listdir(video_directory_cc):
        if video_file.endswith(".mp4"):
            with open(os.path.join(video_directory_cc, video_file), "rb") as video:
                task = twelvelabs_client.embed.task.create(
                    model_name="Marengo-retrieval-2.7",
                    video_file=video
                )
                task.wait_for_done(timeout=1200)
                if task.status == "ready":
                    segments = task.retrieve().video_embedding.segments
                    embeddings = [seg.embeddings_float for seg in segments]
                    start_offsets = [seg.start_offset_sec for seg in segments]
                    end_offsets = [seg.end_offset_sec for seg in segments]
                    video_name = os.path.basename(video_file)
                    video_urls_list = [video_urls_cc.get(video_name, "Unknown")] * len(segments)
                    data = [embeddings, start_offsets, end_offsets, [video_name] * len(segments), video_urls_list]
                    collection.insert(data)

    collection.create_index(
        field_name="embedding",
        index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}}
    )
    collection.load()

def perform_similarity_search_cc(query, collection_name, expected_dim, embedding_model, top_k=3):
    collection = Collection(name=collection_name)
    if embedding_model == "hf":
        query_embedding = hf_embeddings.embed_query(query)
    elif embedding_model == "twelvelabs":
        query_embedding = twelvelabs_client.embed.create(
            model_name="Marengo-retrieval-2.7",
            text=query
        ).text_embedding.segments[0].embeddings_float
    else:
        raise ValueError("Invalid embedding model specified.")

    # Define output fields based on collection type
    output_fields = ["chunk", "file_name", "pdf_url"] if collection_name == pdf_collection_name_cc else ["video_url", "video_name", "start_offset_sec", "end_offset_sec"]

    return collection.search(
        data=[query_embedding[:expected_dim]],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=output_fields
    )

# Retrieve results
def retrieve_results_cc(query):
    pdf_results = perform_similarity_search_cc(query, pdf_collection_name_cc, 384, "hf")
    video_results = perform_similarity_search_cc(query, video_collection_name_cc, 1024, "twelvelabs")

    # Extract relevant PDF details based on the query
    pdf_urls = [
        {
            "chunk": match.entity.get("chunk"),
            "file": match.entity.get("file_name"),
            "pdf_url": match.entity.get("pdf_url"),
        }
        for result in pdf_results for match in result
    ]

    # Extract relevant video details based on the query
    video_urls = [
        {
            "video_url": match.entity.get("video_url"),
            "video_name": match.entity.get("video_name"),
            "start_offset_sec": match.entity.get("start_offset_sec"),
            "end_offset_sec": match.entity.get("end_offset_sec"),
        }
        for result in video_results for match in result
    ]

    # Debugging to confirm results are query-specific
    # print("Query:", query)
    # print("Retrieved PDF URLs:", pdf_urls)
    # print("Retrieved Video URLs:", video_urls)

    return pdf_urls, video_urls


def handle_career_query(query):
    """
    Handles career transition queries using the logic from Code 2.
    """
    print("Routing to Career Transition Logic...")

    def query_llm_cc(query):
        pdf_urls, video_urls = retrieve_results_cc(query)

        system_prompt = (
            "You have to assist people to transition their career paths to desired fields of interest. "
            "Create a structured learning path that will help them successfully transition to their desired career field. "
            "The structured learning path should include courses, video links, related PDF links, and hands-on projects to build skills."
        )
        llm_prompt = f"{system_prompt}\n\nFor the query '{query}', the following relevant resources have been retrieved:\n\n"

        # Add query-specific PDFs
        llm_prompt += "### Top PDFs:\n"
        llm_prompt += "\n".join([
            f"{i+1}. ![PDF Thumbnail](https://via.placeholder.com/150?text=PDF) [{pdf['chunk']}]( {pdf['pdf_url']})"
            for i, pdf in enumerate(pdf_urls)
        ]) + "\n\n"

        # Add query-specific videos
        llm_prompt += "### Top Videos:\n"
        llm_prompt += "\n".join([
            f"{i+1}. ![Video Thumbnail](https://img.youtube.com/vi/{video['video_url'].split('/')[-2]}/0.jpg) "
            f"[Watch {video['video_name']}]( {video['video_url']}) "
            for i, video in enumerate(video_urls)
        ])

        # Debugging to confirm the LLM prompt is built correctly
        # print("LLM Prompt:\n", llm_prompt)

        # LLM Integration
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        response = model.generate_content(llm_prompt)

        # Build the final response text
        response_text = response.text + "\n\nRetrieved PDF Details:\n"
        response_text += "\n".join([
            f"{i+1}. ![PDF Thumbnail](https://via.placeholder.com/150?text=PDF)( {pdf['pdf_url']})"
            for i, pdf in enumerate(pdf_urls)
        ])
        response_text += "\n\nRetrieved Video Details:\n"
        response_text += "\n".join([
            f"{i+1}. ![Video Thumbnail](https://img.youtube.com/vi/{video['video_url'].split('/')[-2]}/0.jpg) [Watch {video['video_name']}]( {video['video_url']})"
            for i, video in enumerate(video_urls)
        ])

        return response_text
    response = query_llm_cc(query) 
    return response



# Main Multi-Agent Logic
def multi_agent_system(query):
    """
    Routes the query to the appropriate agent (Code 1 or Code 2) based on the intent.
    """
    # Classify query intent
    intent = classify_query_intent(query)

    # Route query based on intent
    if intent == "Workplace Stress":
        return handle_workplace_query(query)
    elif intent == "Career Transition":
        return handle_career_query(query)
    else:
        return "Sorry, I couldn't classify your query. Please provide more details."

if __name__ == "__main__":
    # Set the page configuration
    st.set_page_config(page_title="Career and Workplace Assistant", layout="centered")

    # Add logo near the title using columns
    col1, col2 = st.columns([1, 8])  # Adjust column width ratios as needed
    with col1:
        st.image(
            "/Users/mrunmayeerane/Desktop/hackathon/women-in-ai-hackathon/src/data/Images/business-woman-working-on-computer-at-the-desk-cute-cozy-home-workplace-cartoon-style-online-career-self-employed-concept-trendy-modern-illustration-flat-vector.jpg",  # Replace with the path to your logo
            width=85,  # Adjust the size of the logo
        )
    with col2:
        st.title("Career and Workplace Assistant")
    
    # Description below the title
    st.markdown(
        "Welcome to the Career and Workplace Assistant! Ask your query about workplace challenges or career transitions, and we'll guide you with tailored advice and resources."
    )

    # Chat Input
    query = st.text_input("Enter your query:", placeholder="Type your question here...")
    if st.button("Get Advice"):
        if query.strip() == "":
            st.warning("Please enter a query to proceed.")
        else:
            # Process the query through the multi-agent system
            response = multi_agent_system(query)
            # Display the response as markdown
            st.markdown(response, unsafe_allow_html=True)

    # Footer Section
    st.markdown("---")
    st.markdown(
        "Powered by [Streamlit](https://streamlit.io), [LangChain](https://langchain.com)."
    )

# create_pdf_embeddings_ws()
# create_video_embeddings_ws()
# create_pdf_embeddings_cc()
# create_video_embeddings_cc()

# query_1 = "I feel very stressed at work and don't know how to handle it."
# query_2 = "I want to transition my career into Generative AI. Can you help?"

# # Test the system
# response_1 = multi_agent_system(query_1)
# response_2 = multi_agent_system(query_2)

# print("Response to Query 1:\n", response_1)
# print("Response to Query 2:\n", response_2)
