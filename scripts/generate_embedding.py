import os
import django
import json
from sentence_transformers import SentenceTransformer
import sys

project_path = r'C:\Users\user\Desktop\SemanticSearchEngine'
if project_path not in sys.path:
    sys.path.append(project_path)

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'SemanticSearch_engine.settings')
django.setup()

from search.models import FAQ

data_path = r'C:\Users\user\Desktop\SemanticSearchEngine\scripts\Ecommerce_FAQ_Chatbot_dataset.json'

file_path = r'C:\Users\user\Desktop\SemanticSearchEngine\scripts\my_model'
model = SentenceTransformer(file_path)

# Function to generate and save FAQ data
def add_faq_data(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
        
    data = data['questions']
    for item in data:
        question = item['question']
        answer = item['answer']
        
        answer_embedding = model.encode([answer])[0]
        


        # Save the FAQ entry to the database
        faq_entry = FAQ(
            question=question,
            answer=answer,
            answer_embedding=','.join(map(str, answer_embedding)) # Convert to list if it's a numpy array
        )
        faq_entry.save()

    print("Data successfully added to the database.")

add_faq_data(data_path)
