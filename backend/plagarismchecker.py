
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def check_plagiarism(file_path):
	# Load model and tokenizer
	tokenizer = AutoTokenizer.from_pretrained("jpwahle/longformer-base-plagiarism-detection")
	model = AutoModelForSequenceClassification.from_pretrained("jpwahle/longformer-base-plagiarism-detection")

	# Read code file
	with open(file_path, 'r', encoding='utf-8') as f:
		code = f.read()

	# Tokenize input
	inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=4096)

	# Get prediction
	with torch.no_grad():
		outputs = model(**inputs)
		logits = outputs.logits
		pred = torch.argmax(logits, dim=1).item()

	# Return binary result: 1 if AI-generated, 0 if not
	return pred

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print("Usage: python plagarismchecker.py <test.py>")
		sys.exit(1)
	file_path = sys.argv[1]
	result = check_plagiarism(file_path)
	print(result)
