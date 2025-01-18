# SemanticSearch_Django

This project Semantic Search Engine for retrieving frequently asked questions is built using Django, Sentence-Transformers and FAISS. The system allows user to search relevant answers for their queries based on the semantic similarity between the query and FAQ answers. 

### Key Features:
1. Used pretrianed model 'all-MiniLM-L6-v2' from Sentence Tranformer to generate embeddings for FAQ.
2. Utilized FAISS for fast similarity searches over large datasets
3. Integrated with Django to serve the frontend and manage the backend logic.
4. Utilized gemini api for generating JSON dataset including FAQ questions and answers.

### Setup Instructions
1. Created a virtual environment
2. Installed the required libraries and created a Django app and configured superuser.
3. Added a model for FAQ including, questions, answers and answer embeddings.
4. Created two html template for searching and adding FAQ queries
5. Created view for searching and adding and configured the urls.py
6. In the view function, added functionality for getting embeddings of the query and searching using faiss index.
7. Added a jupyter file of model training and model evaluation in scripts folder
8. Used a pretrained model from Hugging face Sentence transformer and trained with a ecommerce FAQ dataset and saved the model
9. Added all the embeddings of the dataset into database by using the model and added the function for it in scripts/generate_embedding.
10. Evaluated the model with MRR and NDCG metrics and added jypyter file in scripts folder
11. Leveraged gemini api to generate FAQ dataset include questions and answers in json format and used this data for evaluation purpose.

    (https://github.com/user-attachments/assets/c30197d8-56cd-40b5-a061-bf4f2ba38981)

