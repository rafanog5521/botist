#!/usr/bin/env python3
import model_interactor
import time
from transformers import pipeline

# Initialize TinyLLAMA pipeline
# pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
def measure_performance(input_text):
    # Start time measurement
    start_time = time.time()
    mi = model_interactor.TinyLlamaModelInteractor(answers=1)

    # Generate response from TinyLLAMA
    response = mi.ask_question(input_text)

    # End time measurement
    end_time = time.time()

    # Calculate response time
    response_time = end_time - start_time
    # Calculate tokens per second
    total_tokens_generated = len(response.split())
    tokens_per_second = total_tokens_generated / response_time

    print("Input text: {}".format(input_text))
    print("Response from TinyLLAMA: {}".format(response))
    print("Response time: {} seconds".format(response_time))
    print("Tokens generated: {}".format(total_tokens_generated))
    print("Tokens per second: {}".format(tokens_per_second))

if __name__ == "__main__":
    # Example input text
    question = "How are you doing today?"

    # Measure performance
    measure_performance(question)
