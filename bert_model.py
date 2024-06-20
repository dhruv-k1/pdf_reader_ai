from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

class BERTQA:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
        self.model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")

    def get_answer(self, question, context):
        inputs = self.tokenizer(question, context, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)

        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
        return answer

bert_qa = BERTQA()
