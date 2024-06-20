from bert_model import preprocess_text

def get_response(model, query):
    query_tokens = preprocess_text(query)
    try:
        responses = model.wv.most_similar(positive=query_tokens, topn=1)
        return responses[0][0] if responses else "Sorry, I don't have an answer for that."
    except KeyError:
        return "Sorry, I don't understand the question."
