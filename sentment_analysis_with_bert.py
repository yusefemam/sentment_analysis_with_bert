#i will use pipline from transformers to use the sentiment analysis model
from transformers import pipeline
import sys


def initialize_model(): #initialize the model
    try:
        return pipeline("sentiment-analysis")
    except Exception as e:
        print(f"Error loading the model: {e}")
        sys.exit(1)


def analyze_sentiment(text, model): #analyze the sentiment
    try:
        result = model(text)[0]
        sentiment = result['label']
        score = result['score']

        emoji = "😊" if sentiment == "POSITIVE" else "😢"
        sentiment_text = "Positive" if sentiment == "POSITIVE" else "Negative"

        return sentiment_text, emoji, score
    except Exception as e:
        return None, None, None


def main():
    print("Welcome to the Sentiment Analysis System!")
    print("Type 'exit' to quit.")

    model = initialize_model()

    while True:
        try:
            name = input("\nEnter your name: ").strip()
            if not name:
                print("Please enter a valid name.")
                continue

            firstname = name.split()[0].capitalize()

            text = input("\nEnter a sentence for sentiment analysis: ").strip()
            if text.lower() == 'exit':
                print(f"\nGoodbye, {firstname}!")
                break
            if not text:
                print("Please enter a valid sentence.")
                continue

            sentiment, emoji, score = analyze_sentiment(text, model) #analyze the sentiment
            if sentiment is None:
                print("An error occurred during analysis. Please try again.")
                continue

            print(
                f"\nResult: {sentiment} {emoji} (Confidence: {score*100:.2f}%)")
            feedback = f"{firstname}, your sentence is good!" if score <= 0.5 else f"{firstname}, you may want to rephrase your sentence."
            print(feedback)

        except KeyboardInterrupt:
            print("\nProgram terminated by user.")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


if __name__ == "__main__": #run the main function
    main()
#made by yusef emam