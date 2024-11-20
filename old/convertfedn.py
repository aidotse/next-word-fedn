import re # AI generated quickly to get the fedn data into readable format
import csv

def read_and_split_fedn():
    # Read the fedn.txt file
    with open('data/fedn.txt', 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Split into sentences (splitting on periods followed by spaces or newlines)
    sentences = re.split(r'(?<=[.!?])\s+', content)
    
    # Split into lines (splitting on newlines)
    lines = content.split('\n')
    
    # Clean up empty strings and whitespace
    sentences = [s.strip() for s in sentences if s.strip()]
    lines = [l.strip() for l in lines if l.strip()]
    
    # Save sentences to fedseq.csv
    with open('data/fedseq.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['sequence_id', 'text'])  # Header
        for idx, sentence in enumerate(sentences):
            writer.writerow([idx + 1, sentence])
    
    return {
        'full_text': content,
        'sentences': sentences,
        'lines': lines
    }

if __name__ == "__main__":
    # Example usage
    result = read_and_split_fedn()
    print(f"Saved {len(result['sentences'])} sentences to fedseq.csv")
    
    # Print first few sentences as example
    print("\nFirst 3 sentences from the data:")
    for i, sentence in enumerate(result['sentences'][:3]):
        print(f"{i+1}. {sentence}")
