
from django.shortcuts import render, redirect
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from search.models import FAQ

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.preprocessing import normalize


# Initialize SentenceTransformer model
file_path = r'C:\Users\user\Desktop\SemanticSearchEngine\scripts\my_model'
model = SentenceTransformer(file_path)


def search(request):
    if request.method == 'POST':
        query = request.POST.get('query')
        if not query:
            return render(request, 'index.html', {'error': 'Please enter a query.'})

        # Retrieve all FAQs from the database
        faqs = FAQ.objects.all()
        if not faqs.exists():
            return render(request, 'index.html', {'error': 'No FAQs found in the database.'})

        # Create a FAISS index
        embeddings = []
        faq_ids = []  # To keep track of FAQ IDs corresponding to embeddings
        for faq in faqs:
            # Ensure that answer_embedding is properly processed as a list of floats
            try:
                embedding_list = list(map(float, faq.answer_embedding.split(',')))
                embeddings.append(np.array(embedding_list, dtype='float32'))
                faq_ids.append(faq.id)
            except AttributeError:
                print(f"Skipping FAQ due to invalid embedding for question: {faq.question}")
                continue

        # Check if embeddings were successfully populated
        if not embeddings:
            return render(request, 'index.html', {'error': 'No valid embeddings found in the database.'})

        # Create the FAISS index
        embedding_dim = len(embeddings[0])  # Assuming all embeddings have the same dimension
        index = faiss.IndexFlatL2(embedding_dim)

        # Add embeddings to the index
        index.add(np.array(embeddings, dtype='float32'))

        # Generate the query embedding
        query_embedding = model.encode([query])[0]
        query_embedding = np.array([query_embedding], dtype='float32')

        # Normalize the query embedding and FAQ embeddings to unit vectors
        query_embedding_normalized = normalize(query_embedding)
        embeddings_normalized = normalize(np.array(embeddings))

        # Search the FAISS index using L2 distance
        k = 5  # Number of results to retrieve
        distances, indices = index.search(query_embedding, k)

        # Calculate cosine similarity using sklearn's cosine_similarity function
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # Ensure the index is valid
                faq_id = faq_ids[idx]
                faq = FAQ.objects.filter(id=faq_id).first()
                if faq:
                    # Calculate cosine similarity between the query and the FAQ embedding
                    similarity = cosine_similarity(query_embedding_normalized, embeddings_normalized[idx].reshape(1, -1))[0][0]
                    if similarity > 0.5:
                        results.append({
                            'question': faq.question,
                            'answer': faq.answer,
                            'similarity': similarity,  # Cosine similarity score
                        })

        return render(request, 'index.html', {'results': results, 'query': query})

    return render(request, 'index.html')




def add_faq(request):
    if request.method == "POST":
        # Retrieve question and answer from the POST request
        question = request.POST.get('question')
        answer = request.POST.get('answer')

        if not question or not answer:
            return render(request, 'addFaq.html', {'error': 'Both question and answer are required.'})

        # Generate the embedding for the answer
        answer_embedding = model.encode([answer])[0]  # Get embedding for the answer
        # Convert the embedding into a string (CSV format)
        answer_embedding_str = ','.join(map(str, answer_embedding))

        # Save the FAQ to the database
        faq = FAQ.objects.create(
            question=question,
            answer=answer,
            answer_embedding=answer_embedding_str
        )

        # Redirect to a confirmation page or display success message
        return render(request, 'addFaq.html', {'faq_added': True})  # Redirect to success page after saving FAQ

    return render(request, 'addFaq.html')
